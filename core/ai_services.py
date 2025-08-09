import google.generativeai as genai
import pinecone
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Initialization ---
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    pinecone_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
except KeyError as e:
    print(f"Error: Environment variable {e} not set. Please check your configuration.")
    # Handle the error appropriately, maybe exit or raise an exception
    raise

# --- AI Models ---
embedding_model = "text-embedding-004"
generative_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Prompts ---
def load_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return "You are a helpful assistant." # Fallback prompt

master_prompt = load_prompt('prompts/master_prompt.txt')
rag_prompt_template = load_prompt('prompts/rag_prompt.txt')


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def get_embedding(text, model=embedding_model):
    """Generates an embedding for the given text."""
    if not text or not isinstance(text, str):
        print("Warning: Embedding requested for empty or invalid text.")
        # Return a zero-vector or handle as appropriate for your application
        return [0.0] * 768 # Assuming embedding dimension is 768 for text-embedding-004
    text = text.replace("\n", " ")
    return genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")["embedding"]


def process_and_structure_data(text_content):
    """Processes raw text with Gemini to structure it into the UKS format."""
    prompt = f"{master_prompt}\n\n--- Raw Text ---\n{text_content}"
    response = generative_model.generate_content(prompt)

    try:
        # Extract the structured text part from the model's response
        # This assumes the model reliably returns the text in the requested format.
        structured_text = response.text.strip()

        # Safely create the final JSON object
        # This removes the burden of generating valid JSON from the LLM
        lines = structured_text.split('\n')
        uks_data = {}
        for line in lines:
            if 'Title:' in line:
                uks_data['Title'] = line.split(':', 1)[1].strip()
            elif 'Summary:' in line:
                uks_data['Summary'] = line.split(':', 1)[1].strip()
            elif 'Domain:' in line:
                uks_data['Domain'] = line.split(':', 1)[1].strip()
            elif 'Tags:' in line:
                # Handle potential empty tags
                tags_str = line.split(':', 1)[1].strip()
                uks_data['Tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()] if tags_str else []

        # Validate that essential fields are present
        if not all(key in uks_data for key in ['Title', 'Summary', 'Domain', 'Tags']):
             raise ValueError("Model output did not contain all required fields.")

        return uks_data

    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error processing model output: {e}")
        print(f"Raw response was: {response.text}")
        return None


def upsert_to_pinecone(item_id, text_to_embed, metadata):
    """Generates embedding and upserts a vector to Pinecone."""
    if not text_to_embed:
        print(f"Skipping upsert for ID {item_id} due to empty content.")
        return
    vector = get_embedding(text_to_embed)
    pinecone_index.upsert(vectors=[{'id': item_id, 'values': vector, 'metadata': metadata}])
    print(f"Successfully upserted ID: {item_id}")


# NEW FUNCTION FOR PHASE 3
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def query_knowledge_base(query, top_k=5):
    """
    Performs the RAG process:
    1. Embeds the user query.
    2. Searches Pinecone for relevant context.
    3. Constructs a prompt with the context.
    4. Generates a final answer using the generative model.
    """
    # 1. Embed the user query
    query_vector = get_embedding(query, model="RETRIEVAL_QUERY")

    # 2. Search Pinecone for relevant context
    search_results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 3. Construct the context from search results
    context = ""
    if search_results['matches']:
        for match in search_results['matches']:
            metadata = match['metadata']
            # We use a compact representation of the knowledge for the context
            context += f"Title: {metadata.get('Title', 'N/A')}\n"
            context += f"Summary: {metadata.get('Summary', 'N/A')}\n\n"
    
    if not context:
        # Handle cases where no relevant documents are found
        # You could return a predefined message or let Gemini handle it
        print("No relevant context found in the knowledge base.")
        # Fallback to a simple generation without context
        final_prompt = f"The user asked: '{query}'. I couldn't find anything relevant in my knowledge base. Please answer the question based on your general knowledge, but mention that you didn't find specific information in the user's 'Second Brain'."

    else:
        # 4. Construct the final prompt using the RAG template
        final_prompt = rag_prompt_template.format(user_query=query, context=context)

    # 5. Generate a final answer
    final_response = generative_model.generate_content(final_prompt)
    
    return final_response.text
