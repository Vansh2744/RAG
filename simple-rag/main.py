import os
import io
import base64
import fitz  # PyMuPDF
from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


IMAGE_DIR = "images"
MAX_IMAGES_IN_PROMPT = 3  # Groq limits images per request
os.makedirs(IMAGE_DIR, exist_ok=True)

embeddings = CohereEmbeddings(model="embed-v4.0")
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=os.getenv('GROQ_API_KEY'))

store = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def save_as_jpeg(img_bytes: bytes, path: str):
    """Normalize any PDF image to a small RGB JPEG."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((1024, 1024))
    img.save(path, "JPEG", quality=85)


def image_block(path: str) -> dict:
    """Turn an image file into a LangChain multimodal content block."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def caption_image(path: str) -> str:
    msg = HumanMessage(content=[
        {"type": "text", "text": (
            "Describe this image for a search index. Include all visible "
            "text, numbers, labels, and what any chart or diagram shows."
        )},
        image_block(path),
    ])
    return llm.invoke([msg]).content

def ingest_pdf(pdf_path: str):
    pdf = fitz.open(pdf_path)
    name = os.path.basename(pdf_path)
    docs, ids = [], []

    for page_num, page in enumerate(pdf, start=1):
        # 1) text chunks
        for i, chunk in enumerate(splitter.split_text(page.get_text())):
            ids.append(f"{name}-p{page_num}-t{i}")
            docs.append(Document(
                page_content=chunk,
                metadata={"type": "text", "source": name, "page": page_num},
            ))

        # 2) images -> save -> caption -> index the caption
        for i, img in enumerate(page.get_images(full=True)):
            doc_id = f"{name}-p{page_num}-i{i}"
            if store.get(ids=[doc_id])["ids"]:
                continue  # already indexed, skip the paid vision call

            data = pdf.extract_image(img[0])
            if len(data["image"]) < 5000:  # skip icons and logos
                continue

            img_path = f"{IMAGE_DIR}/{doc_id}.jpg"
            save_as_jpeg(data["image"], img_path)

            ids.append(doc_id)
            docs.append(Document(
                page_content=caption_image(img_path),
                metadata={"type": "image", "source": name, "page": page_num,
                          "image_path": img_path},
            ))

    store.add_documents(docs, ids=ids)
    print(f"Indexed {len(docs)} items from {name}")

def ask(question: str, k: int = 5) -> str:
    hits = store.as_retriever(search_kwargs={"k": k}).invoke(question)

    content, images_used = [], 0
    for d in hits:
        m = d.metadata
        label = f"[{m['type']}, {m['source']} p.{m['page']}]"

        if m["type"] == "image" and images_used < MAX_IMAGES_IN_PROMPT:
            content.append({"type": "text", "text": label})
            content.append(image_block(m["image_path"]))  # the REAL image
            images_used += 1
        else:
            # text hit, or an image over the cap (caption is the fallback)
            content.append({"type": "text", "text": f"{label}\n{d.page_content}"})

    content.append({"type": "text", "text": (
        "Answer using only the context above. Cite the source and page. "
        f"If the answer isn't there, say so.\n\nQuestion: {question}"
    )})

    return llm.invoke([HumanMessage(content=content)]).content

if __name__ == "__main__":
    ingest_pdf("dummy_company_sales_report.pdf")   # run once; comment out afterwards
    print(ask("which month having the most profit"))