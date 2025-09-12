import pandas as pd
from langchain_ollama import OllamaLLM
from vector import retriever
from langchain_core.prompts import ChatPromptTemplate
import time

df = pd.read_excel("Data/vegetation_indices_predictions_ALL_IMAGES.xlsx")

model = OllamaLLM(model="gemma3:1b")

chat_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert agricultural advisor. "
     "Use vegetation indices, Analyse them to give practical recommendations."
     "You should give recommendation on agricultural aspect but it should be scientific and technical."
     "Also provide a priority based ranking for your suggestions."),
    ("user",
     "Context from knowledge base:\n{context}\n\n"
     "Vegetation Index values:\nVNDVI:{vndvi}, MGRVI:{mgrvi}, VARI:{vari}\n\n"
     "Give a clear recommendation for farmers.")
])

chain = chat_template | model
results = []
i = 0

for _, row in df.iterrows():
    query = f"Recommendations for VNDVI={row['predicted_vNDVI']}, MGRVI={row['predicted_MGRVI']}, VARI={row['predicted_VARI']}"
    start = time.time()
    docs = retriever.invoke(query)
    print("Retriever took:", time.time() - start, "seconds")
    context = "\n".join([d.page_content for d in docs])

    if i == 2:
        break
    i += 1

    start = time.time()
    result = chain.invoke({
        "context": context,
        "vndvi": row["predicted_vNDVI"],
        "mgrvi": row["predicted_MGRVI"],
        "vari": row["predicted_VARI"]
    })
    print("LLM took:", time.time() - start, "seconds")

    recommendation = result.content if hasattr(result, "content") else str(result)

    results.append({
        "image_id": row["image_id"],
        "recommendation": recommendation
    })

df2 = pd.DataFrame(results)
df2.to_csv("Data/Output.csv", index=False)
