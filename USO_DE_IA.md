# USO DE INTELIGENCIA ARTIFICIAL

> ** Documentación del uso de IA**  
> Este archivo documenta el uso de herramientas
> de Inteligencia Artificial durante el desarrollo de este proyecto.

---

## 1. Herramientas de IA Utilizadas

| Herramienta | Tareas en las que se usó                                                           |
|-------------|------------------------------------------------------------------------------------|
| Claude      | _Estructuración de Plotly Dash, lógica de APIs y control de excepciones._          |
| GitHub Copilot  | _Consulta de redacción de mensajes de commit y textos informativos del dashboard._ |
| Gemini     | _Sugerencias de estructuras de documentación_                                      |

---
## 2. Ejemplos de Prompts Utilizados

### Prompt 1
**Herramienta:** _Claude_  
**Contexto / tarea:** _Desarrollar una interfaz multipágina en Plotly Dash  <br> con estilos CSS externos._
```
Se tiene esta imagen de la estructura de trabajo para el siguiente proyecto.  Se requiere realizar un Dashboard interactivo funcional con Plotly Dash para mostrar Gráficos profesionales. Exploración visual de embeddings. Brindame una estructura editable que sea con un Diseño visual profesional y navegable para trabajar dentro de mi proyecto
```

**Resultado obtenido:** _Código base de un dashboard con ploty dash_  

---
### Prompt 2
**Herramienta:** _Claude_  
**Contexto / tarea:** _Validación de scraper de letras musicales_
```
dime si este codigo "", cumple con esto "Web Scraping y Enriquecimiento del Corpus: Implementar un scraper funcional que extraiga letras musicales de al menos un sitio web, respetando buenas prácticas (rate limiting, robots.txt),"
```
**Resultado obtenido:** _Veredicto: Cumple con el criterio<br>
El único punto débil es la ausencia de verificación explícita de robots.txt en el código<br>
Recomendación: agregar verificación explícita con urllib.robotparser para cubrir robots.txt formalmente_

---

### Prompt 3
**Herramienta:** _Claude_  
**Contexto / tarea:** _Evaluar error de plotly en la propiedad colorscale_
```
que significa este error "ValueError:

Invalid value of type 'builtins.list' received for the 'colorscale' property of bar.marker

Received value: [[0, '#7C3AED60'], [1, '#00D4FF']]
```

**Resultado obtenido:** _El error dice que bar.marker.colorscale no acepta tu lista. Esto sucede porque, en los gráficos de barras (go.Bar), la propiedad colorscale solo funciona si le dices a Plotly qué valores numéricos debe usar para mapear esos colores.

Si no tienes una lista de números asignada a la propiedad color, Plotly no sabe cómo aplicar una "escala" y, por lo tanto, rechaza la configuración.._

---

## 3. Reflexión sobre el Aprendizaje

_El uso de IA (Claude) me permitió acelerar el desarrollo del proyecto, especialmente en la creación de la arquitectura multipágina en Dash._

_No obstante, algunas respuestas requirieron diversos ajustes para adaptarse a la implementación específica, ya que no siempre contemplaba detalles como variables globales o dependencias entre callbacks. Fue necesario aplicar criterio propio para depurar errores y validar correctamente los resultados antes de habilitar las páginas. La IA facilitó el proceso, pero la integración final demandó análisis técnico y varias pruebas._

---

## 4. Modificaciones al Código / Análisis Generado por IA

- **Modificación 1:** _Se reorganizan variables globales para evitar conflictos entre hilos y callbacks._
- **Modificación 2:** _Se agregan validaciones adicionales para impedir que las páginas de visualización se habiliten si el DataFrame del pipeline es None o si ocurrió una excepción._
- **Modificación 3:**  _Se implementó la verificación de robots.txt mediante urllib.robotparser para asegurar un scraping ético, siguiendo la recomendación de la IA."
Esto cerraría el ciclo entre lo que la IA sugirió y lo que realmente hiciste._
---
*Última actualización: Jueves 26 de marzo del 2026*