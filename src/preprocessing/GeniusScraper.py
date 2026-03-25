import os
import re
import time
import random
import pandas as pd
import musicbrainzngs
import lyricsgenius
from tqdm import tqdm
import urllib.robotparser
musicbrainzngs.set_useragent("GeniusScraper", "1.0", "scraper@example.com")

# ── Token ──────────────────────────────────────────────────────────────────────
GENIUS_TOKEN = "vDJynfsZUKxymbrua_RmHLGWtgDorffPH0LMttrwr-bigxnAc19htWL2t5tRxLmg"

# ── Artistas a buscar ──────────────────────────────────────────────────────────
ARTISTAS = [
    "Drake", "Taylor Swift", "Ed Sheeran", "Ariana Grande", "Billie Eilish",
    "The Weeknd", "Post Malone", "Kendrick Lamar", "Beyoncé", "Rihanna",
    "Eminem", "Jay-Z", "Kanye West", "Lil Wayne", "Nicki Minaj",
    "Bruno Mars", "Justin Bieber", "Lady Gaga", "Katy Perry", "Adele",
    "Harry Styles", "Olivia Rodrigo", "Dua Lipa", "Lizzo", "Cardi B",
    "Travis Scott", "J. Cole", "21 Savage", "Future", "Lil Baby",
    "Juice WRLD", "XXXTentacion", "Pop Smoke", "Roddy Ricch", "DaBaby",
    "Bad Bunny", "J Balvin", "Ozuna", "Maluma", "Daddy Yankee",
    "Shakira", "Luis Fonsi", "Enrique Iglesias", "Marc Anthony", "Romeo Santos",
    "Bob Dylan", "Bruce Springsteen", "Johnny Cash", "Willie Nelson", "Dolly Parton",
    "Metallica", "AC/DC", "Led Zeppelin", "Pink Floyd", "The Beatles",
    "Michael Jackson", "Prince", "Madonna", "Whitney Houston", "Mariah Carey",
    "Frank Sinatra", "Aretha Franklin", "Ray Charles", "James Brown", "Stevie Wonder",
    "John Legend", "Sam Smith", "Shawn Mendes", "Niall Horan", "Charlie Puth",
    "Halsey", "Selena Gomez", "Demi Lovato", "Miley Cyrus", "Camila Cabello",
    "SZA", "H.E.R.", "Summer Walker", "Jhené Aiko", "Ella Mai",
    "Coldplay", "Imagine Dragons", "Twenty One Pilots", "Panic! At The Disco", "Fall Out Boy",
    "Green Day", "Linkin Park", "Foo Fighters", "Red Hot Chili Peppers", "Radiohead",
    "Chance the Rapper", "childish Gambino", "Tyler the Creator", "Frank Ocean", "Anderson .Paak",
    "Megan Thee Stallion", "Doja Cat", "Saweetie", "City Girls", "Rico Nasty",
    "Lil Nas X", "Jack Harlow", "Polo G", "NBA YoungBoy", "King Von",
    "Morgan Wallen", "Luke Combs", "Blake Shelton", "Tim McGraw", "Garth Brooks",
    "Alicia Keys", "Mary J. Blige", "Ne-Yo", "Usher", "R. Kelly",
    "Chris Brown", "Trey Songz", "Tank", "Ginuwine", "Keith Sweat",
    "Bon Jovi", "U2", "The Rolling Stones", "Aerosmith", "Guns N' Roses",
]


class GeniusScraper:
    """
    Scraper para Genius usando lyricsgenius.
    Extrae: artist, title, lyric, year, genre
    """

    def __init__(self, token, delay=1.5, checkpoint_file="genius_dataset.csv"):
        self.delay           = delay
        self.checkpoint_file = checkpoint_file
        self.data            = []

        # Inicializar lyricsgenius — maneja URLs, redirecciones y limpieza
        self.genius = lyricsgenius.Genius(
            token,
            sleep_time=delay,
            timeout=15,
            retries=3,
            verbose=False,          # silenciar logs internos
            remove_section_headers=False,  # mantener [Verse], [Chorus] etc.
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Demo)", "(Instrumental)"],
        )
        print(f"  [Genius] Inicializado con lyricsgenius. Delay={delay}s")


    # ── Verificar Robot ────────────────────────────────────────────────────────────
    def _check_robots(self,url_base, user_agent='*'):
        """Verifica rutas clave de Genius según robots.txt. Retorna True si el scraping está permitido."""
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{url_base}/robots.txt")
        try:
            rp.read()
            print("  [ROBOTS] robots.txt leído correctamente")
            test_paths = ["/", "/search", "/api", "/sitemap.xml"]
            bloqueadas = 0
            for path in test_paths:
                allowed = rp.can_fetch(user_agent, url_base + path)
                status = "[OK]" if allowed else "[BLOQUEADO]"
                print(f"    {status}: {path}")
                if not allowed:
                    bloqueadas += 1
            return bloqueadas == 0
        except Exception as e:
            print(f"  [ROBOTS] Error leyendo robots.txt: {e}.")
            return False
    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return set()
        try:
            df   = pd.read_csv(self.checkpoint_file, usecols=["artist"], encoding="utf-8-sig")
            done = set(df["artist"].dropna().unique())
            print(f"  [CHECKPOINT] {len(done)} artistas ya procesados, se omitirán")
            return done
        except Exception:
            return set()

    def _save_checkpoint(self, canciones):
        df_new    = pd.DataFrame(canciones)
        write_hdr = not os.path.exists(self.checkpoint_file)
        df_new.to_csv(
            self.checkpoint_file, mode="a",
            index=False, header=write_hdr, encoding="utf-8-sig"
        )

    # ── Limpieza de letra ─────────────────────────────────────────────────────

    def _clean_lyric(self, raw):
        """Limpia metadata residual que lyricsgenius no elimina."""
        if not raw:
            return None

        # Encontrar primer [sección] y tomar desde ahí
        match = re.search(r'\[', raw)
        if match:
            raw = raw[match.start():]

        # Eliminar líneas de metadata
        clean = []
        for line in raw.split("\n"):
            line_s = line.strip()
            if re.match(
                r"^(\d+\s+Contributors?|Translations|You might also like|"
                r"Español|Português|Français|Deutsch|Italiano|Türkçe|"
                r"Русский|Polski|English|Read More\.?)$",
                line_s
            ):
                continue
            clean.append(line_s)

        lyric = "\n".join(clean).strip()
        lyric = re.sub(r"\n{3,}", "\n\n", lyric)
        return lyric if len(lyric) > 50 else None

    # ── Año y género (MusicBrainz) ────────────────────────────────────────────

    def _get_year_genre(self, artist_name, title):
        try:
            result = musicbrainzngs.search_recordings(
                recording=title, artist=artist_name, limit=1
            )
            recs = result.get("recording-list", [])
            if not recs:
                return None, None
            rec  = recs[0]
            year = None
            rels = rec.get("release-list", [])
            if rels:
                date = rels[0].get("date", "")
                year = date[:4] if date else None
            genre = None
            tags  = rec.get("tag-list", [])
            if tags:
                genre = sorted(
                    tags, key=lambda t: int(t.get("count", 0)), reverse=True
                )[0].get("name")
            return year, genre
        except Exception:
            return None, None

    # ── Scraping por artista ──────────────────────────────────────────────────

    def scrape_artist(self, artist_name, max_songs=10):
        """Obtiene canciones de un artista usando lyricsgenius."""
        canciones = []
        try:
            # lyricsgenius busca el artista y sus canciones automáticamente
            artist = self.genius.search_artist(
                artist_name,
                max_songs=max_songs,
                sort="popularity",
            )
            if not artist or not artist.songs:
                return canciones

            for song in artist.songs:
                lyric = self._clean_lyric(song.lyrics)
                if not lyric:
                    continue

                # Año desde lyricsgenius o MusicBrainz como fallback
                year = None
                if hasattr(song, 'year') and song.year:
                    year = str(song.year)

                year_mb, genre = self._get_year_genre(artist_name, song.title)
                if not year:
                    year = year_mb

                canciones.append({
                    "artist": artist_name,
                    "title":  song.title,
                    "lyric":  lyric,
                    "year":   year,
                    "genre":  genre,
                })

        except Exception as e:
            print(f"  [ERROR] {artist_name}: {e}")

        return canciones

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self, artistas=None, max_songs=10):
        """
        Pipeline completo con checkpoint.
        artistas: lista de nombres de artistas a procesar
        """
        if artistas is None:
            artistas = ARTISTAS

        print("=" * 55)
        print("  Genius Scraper  |  lyricsgenius + Checkpoint")
        print(f"  Artistas: {len(artistas)} | Canciones/artista: {max_songs}")
        print("=" * 55)

        if not self.check_robots("https://genius.com"):
            print("  [ROBOTS] Acceso restringido por robots.txt. Abortando.")
            return pd.DataFrame()



        done       = self._load_checkpoint()
        pendientes = [a for a in artistas if a not in done]
        print(f"\nTotal: {len(artistas)} | Pendientes: {len(pendientes)}\n")

        for i, nombre in enumerate(tqdm(pendientes, desc="Artistas"), 1):
            canciones = self.scrape_artist(nombre, max_songs=max_songs)

            if canciones:
                self.data.extend(canciones)
                self._save_checkpoint(canciones)
                print(f"  [{i}] {nombre} → {len(canciones)} canciones OK")
            else:
                print(f"  [{i}] {nombre} → sin canciones")

            # Pausa cada 10 artistas
            if i % 10 == 0:
                pausa = random.uniform(10, 20)
                print(f"\n  [PAUSA] {pausa:.0f}s tras {i} artistas...\n")
                time.sleep(pausa)

        # Cargar dataset completo
        if os.path.exists(self.checkpoint_file):
            self.df = pd.read_csv(self.checkpoint_file, encoding="utf-8-sig")
        else:
            self.df = pd.DataFrame(self.data)

        print("\n" + "=" * 55)
        print(f"  Total canciones : {len(self.df)}")
        if not self.df.empty and "artist" in self.df.columns:
            print(f"  Total artistas  : {self.df['artist'].nunique()}")
            print(f"  Con año         : {self.df['year'].notna().sum()}")
            print(f"  Con género      : {self.df['genre'].notna().sum()}")
        print(f"  Guardado en     : {self.checkpoint_file}")
        print("=" * 55)
        return self.df


