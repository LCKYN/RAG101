import os

from anthropic import Anthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = (
    ""
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid the tokenizers warning


class RAGSystem:
    def __init__(self, model_name="claude-3-sonnet-20240229"):
        # Initialize Anthropic client
        self.client = Anthropic()
        self.model_name = model_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Load and process documents
        loader = TextLoader("path_to_your_document.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

        # Initialize conversation history and system prompt
        self.conversation_history = []
        self.system_prompt = "Answer shortest as possible."

    def retrieve_relevant_docs(self, query: str, k: int = 3) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def generate_response(self, query: str) -> str:
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Prepare the messages for the API call
        messages = self.conversation_history.copy()
        # Add the user's query
        messages.append(
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        )

        # Generate response using Anthropic API
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=self.system_prompt,  # Set the system prompt as a top-level parameter
            messages=messages,
        )

        ai_response = response.content[0].text

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        # Keep only the last 10 messages (5 exchanges) in the history
        self.conversation_history = self.conversation_history[-10:]

        return ai_response


# Example usage
rag = RAGSystem()

# Prime the system with an example
example_prompt = "hi my name is Jeam"
example_response = rag.generate_response(example_prompt)
print(f"Claude: {example_response}")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = rag.generate_response(user_input)
    print(f"Claude: {response}")
