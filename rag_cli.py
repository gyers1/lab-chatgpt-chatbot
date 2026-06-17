from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
DEFAULT_DOCS_DIR = Path("data/sample_docs")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """아래 '문맥'만 근거로 한국어로 답하세요.
문맥에 답이 없으면 반드시 "문서에서 찾을 수 없습니다"라고만 답하세요.
이전 대화는 질문의 맥락을 이해하는 데만 참고하고, 사실 판단은 반드시 문맥으로만 하세요.

[문맥]
{context}

[이전 대화]
{history}

[질문]
{question}
"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="멀티턴 RAG CLI. 문서 내용을 검색하고 이전 대화를 반영해 답합니다."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=[str(DEFAULT_DOCS_DIR)],
        help="문서 파일 또는 문서 폴더 경로들 (.md / .txt / .pdf)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="검색할 문서 청크 개수",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="질문 재작성에 사용할 최근 대화 턴 수",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="문서 청크 크기",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="문서 청크 겹침 크기",
    )
    return parser.parse_args()


def collect_document_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in items:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"문서를 찾을 수 없습니다: {path}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    paths.append(child)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
        else:
            raise ValueError(f"지원하지 않는 문서 형식입니다: {path}")

    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise ValueError("인덱싱할 문서를 찾지 못했습니다.")
    return unique_paths


def load_documents(paths: list[Path]):
    docs = []
    for path in paths:
        loader = PyMuPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_retriever(paths: list[Path], chunk_size: int, chunk_overlap: int, top_k: int):
    docs = load_documents(paths)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(docs)

    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def history_to_text(history: list[dict[str, str]], turns: int) -> str:
    if not history:
        return "없음"

    recent = history[-turns * 2 :]
    lines: list[str] = []
    for message in recent:
        role = "사용자" if message["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def format_docs(docs) -> str:
    parts: list[str] = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "")).name or "unknown"
        parts.append(f"[출처: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(answer_llm: ChatOpenAI, retriever, question: str, history_text: str) -> str:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = ANSWER_PROMPT | answer_llm | StrOutputParser()
    return chain.invoke(
        {
            "context": context,
            "history": history_text,
            "question": question,
        }
    )


def print_help() -> None:
    print("명령어:")
    print("/history  - 현재 누적 대화 보기")
    print("/reset    - 대화 초기화")
    print("/exit     - 종료")
    print("/quit     - 종료")


def main() -> None:
    load_dotenv()
    args = parse_args()

    doc_paths = collect_document_paths(args.docs)
    retriever = build_retriever(
        doc_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )

    answer_llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

    history: list[dict[str, str]] = []

    print("=== 멀티턴 RAG CLI ===")
    print("로드된 문서:")
    for path in doc_paths:
        print(f"- {path}")
    print_help()
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("종료")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/exit", "/quit"}:
            print("종료")
            break
        if lowered == "/reset":
            history.clear()
            print("(대화 초기화 완료)")
            continue
        if lowered == "/history":
            if not history:
                print("(현재 히스토리가 없습니다)")
                continue
            for idx, message in enumerate(history, start=1):
                print(f"[{idx}] {message['role']}: {message['content']}")
            continue

        history.append({"role": "user", "content": user_input})
        history_text = history_to_text(history, args.history_turns)

        # 검색은 원문 질문으로만 하고, 이전 대화는 답변 생성 단계에서만 반영한다.
        answer = answer_question(answer_llm, retriever, user_input, history_text)
        history.append({"role": "assistant", "content": answer})

        print("RAG:", answer)


if __name__ == "__main__":
    main()
