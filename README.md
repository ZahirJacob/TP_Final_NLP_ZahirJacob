# Asistente RAG multifuente con agente ReAct

Trabajo Práctico Final de la asignatura **Procesamiento del Lenguaje Natural**. Consiste en un asistente conversacional para una empresa de electrodomésticos que responde consultas combinando tres fuentes de información heterogéneas: documentos de texto, datos tabulares y un grafo de conocimiento. El sistema se ejecuta íntegramente con modelos locales, sin depender de APIs externas.

## Arquitectura

La consulta del usuario atraviesa las siguientes etapas:

1. **Clasificación de intención.** Determina qué fuente debe consultarse (`vectorial`, `tabular` o `grafo`). Se comparan dos enfoques: un clasificador supervisado (TF-IDF + regresión logística) y un LLM local con *few-shot prompting*.
2. **Recuperación según la fuente.**
   - *Vectorial:* búsqueda híbrida (semántica en ChromaDB + BM25) seguida de un *re-ranking* mediante un modelo *cross-encoder*.
   - *Tabular:* el LLM traduce la consulta a filtros JSON que se aplican sobre un DataFrame de productos.
   - *Grafo:* el LLM genera consultas Cypher que se ejecutan sobre GrafitoDB.
3. **Generación de la respuesta.** El LLM redacta la respuesta final a partir del contexto recuperado, con memoria conversacional de los últimos turnos.
4. **Agente ReAct.** Como extensión, un agente selecciona de forma autónoma entre cuatro herramientas (`doc_search`, `table_search`, `graph_search` y `analytics_tool`), ejecuta la acción correspondiente, observa el resultado y elabora la respuesta.

## Fuentes de información

| Fuente | Contenido | Almacenamiento |
|---|---|---|
| Preguntas frecuentes, manuales, reseñas y *tickets* de soporte | ~10.000 documentos, segmentados en ~11.000 fragmentos | ChromaDB + índice BM25 |
| Catálogo de productos | Precio, stock, marca, categoría, voltaje y garantía | Pandas |
| Relaciones producto–categoría–marca | Nodos y aristas derivados del catálogo | GrafitoDB |

## Tecnologías

- **Modelos:** `Qwen/Qwen2.5-3B-Instruct` (generación), `intfloat/multilingual-e5-small` (*embeddings*), *cross-encoder* para *re-ranking*.
- **Recuperación:** ChromaDB, txtai (BM25), LangChain (segmentación y herramientas).
- **Datos:** Pandas, GrafitoDB, scikit-learn.
- **Entorno:** Python, Google Colab con GPU.

## Resultados

Los clasificadores de intención se evalúan sobre un conjunto sintético de 90 consultas (27 reservadas para prueba), que incluye un subconjunto de consultas más exigentes: lenguaje informal, errores de tipeo, ausencia de signos de pregunta y vocabulario que mezcla distintas fuentes.

| Clasificador | Exactitud | F1 macro |
|---|---|---|
| TF-IDF + regresión logística | 0,85 | 0,85 |
| LLM local con *few-shot prompting* | 0,96 | 0,96 |

Los errores del modelo supervisado se concentran en consultas ambiguas que combinan vocabulario de dos fuentes; el LLM las resuelve mejor, por lo que es el clasificador adoptado en el *pipeline* principal. Cabe señalar que el conjunto sigue siendo reducido, por lo que los resultados deben interpretarse como una validación funcional y no como una medida de generalización.

Las respuestas generadas se apoyan exclusivamente en el contexto recuperado e incorporan opiniones mixtas de los usuarios cuando corresponde, sin introducir información externa.

## Contenido del repositorio

- `TP_NLP_ZAHIR_JACOB.ipynb`: *notebook* con la implementación completa y las salidas de ejecución.
- `INFORME_TP_FINAL_NLP_ZAHIR_JACOB.pdf`: informe con el diseño, las decisiones tomadas y el análisis de resultados.

## Ejecución

El *notebook* está preparado para Google Colab. Las dependencias se instalan en la primera celda y el conjunto de datos se descarga automáticamente desde Google Drive. Se recomienda un entorno con GPU para la carga del modelo de lenguaje.

## Autor

Zahir Jacob
