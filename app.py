from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import logging
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import regex as re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentQASystem:
    def __init__(self):
        self.docsearch = None
        self.llm = None
        self.chain = None
        self.text_chunks = []
        self.sections_metadata = []
        self.embeddings = None
        self.initialize_system()

    def load_json(self, file_path):
        """Load and process JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            texts = []
            metadata = []
            for chapter in data:
                for subchapter in chapter.get('subchapters', []):
                    for section in subchapter.get('sections', []):
                        text = section.get('text', '')
                        if text:
                            texts.append(text)
                            metadata.append({
                                'chapterNum': chapter.get('chapterNum'),
                                'subChapterNum': subchapter.get('subChapterNum'),
                                'sectionNum': section.get('sectionNum'),
                                'title': section.get('title')
                            })
            return texts, metadata
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return [], []

    def clean_text(self, chunks):
        """Clean and normalize text chunks"""
        cleaned_chunks = []
        for chunk in chunks:
            chunk = re.sub(r"\s*\.\s*", " ", chunk)  # Replace dots with spaces
            chunk = re.sub(r"[^\x20-\x7E\u0400-\u04FF]+", " ", chunk)  # Remove non-printable
            chunk = chunk.strip()
            if chunk:
                cleaned_chunks.append(chunk)
        return cleaned_chunks

    def initialize_system(self):
        """Initialize the QA system with JSON data"""
        try:
            json_path = 'OREC/cleaned_data.json'
            if not os.path.exists(json_path):
                logger.error(f"JSON file not found: {json_path}")
                return False

            texts, metadata = self.load_json(json_path)
            if not texts:
                logger.error("No text extracted from JSON")
                return False

            # Split into chunks
            char_text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            self.text_chunks = []
            self.sections_metadata = []
            for text, meta in zip(texts, metadata):
                chunks = char_text_splitter.split_text(text)
                cleaned_chunks = self.clean_text(chunks)
                self.text_chunks.extend(cleaned_chunks)
                self.sections_metadata.extend([meta] * len(cleaned_chunks))

            if not self.text_chunks:
                logger.error("No text chunks created")
                return False

            # Initialize embeddings and vector store
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.docsearch = FAISS.from_texts(self.text_chunks, self.embeddings, metadatas=self.sections_metadata)

            # Initialize LLM and QA chain
            self.llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, max_tokens=500)
            self.chain = load_qa_chain(self.llm, chain_type="stuff")

            logger.info(f"System initialized with {len(self.text_chunks)} text chunks")
            return True
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False

    def get_answer(self, query, k=4):
        """Get answer for a query with source tracking"""
        try:
            if not self.docsearch or not self.chain:
                return {
                    'answer': 'System not properly initialized.',
                    'source': None,
                    'error': True
                }

            # Get similar documents
            docs = self.docsearch.similarity_search(query, k=k)
            if not docs:
                return {
                    'answer': 'No relevant information found in the document.',
                    'source': None,
                    'no_docs': True
                }

            # Enhanced prompt
            enhanced_query = f"""
            Based on the OREC Code and Rule Book, answer: {query}
            - Provide a direct, comprehensive answer based only on the document.
            - If no information is available, state: "I don't have information about this."
            - Include specific details like timeframes, amounts, or conditions.
            """

            with get_openai_callback() as cb:
                response = self.chain.run(input_documents=docs, question=enhanced_query)
                logger.info(f"Query cost: {cb}")

            # Check for no-info response
            no_info_indicators = ["i don't have", "not mentioned", "not specified", "not provided"]
            if any(indicator in response.lower() for indicator in no_info_indicators):
                return {
                    'answer': 'I don\'t have specific information about that topic in the OREC Code and Rule Book.',
                    'source': None,
                    'no_answer': True
                }

            # Get most relevant section (first document's metadata)
            source = docs[0].metadata if docs else None

            return {
                'answer': response.strip(),
                'source': source,
                'error': False
            }
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            return {
                'answer': 'Error processing your question.',
                'source': None,
                'error': True
            }

    def get_document_sections(self):
        """Get structured document sections for frontend display"""
        try:
            # Load JSON directly to maintain hierarchy
            with open('OREC/cleaned_data.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            logger.error(f"Error getting sections: {e}")
            return []


# Initialize the QA system
qa_system = DocumentQASystem()


@app.route('/')
def index():
    """Serve the main chat interface"""
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>OREC Assistant</title></head>
        <body>
            <h1>Error</h1>
            <p>Cannot find chat.html in templates folder.</p>
            <p>Ensure chat.html is in the 'templates' directory.</p>
        </body>
        </html>
        '''


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system_initialized': qa_system.docsearch is not None,
        'chunks_loaded': len(qa_system.text_chunks)
    })


@app.route('/api/sections', methods=['GET'])
def get_sections():
    """Get document sections for frontend display"""
    try:
        sections = qa_system.get_document_sections()
        return jsonify({'sections': sections, 'total': len(sections)})
    except Exception as e:
        logger.error(f"Error getting sections: {e}")
        return jsonify({'error': 'Failed to load document sections'}), 500


@app.route('/api/query', methods=['POST'])
def query_document():
    """Main endpoint for document queries"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400

        result = qa_system.get_answer(query)
        response = {
            'answer': result['answer'],
            'source': result['source'],
            'type': 'success' if not result.get('error') and not result.get('no_answer') else 'no_answer' if result.get(
                'no_answer') else 'error'
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': 'Internal server error', 'type': 'error'}), 500


if __name__ == '__main__':
    if qa_system.docsearch is None:
        print("❌ Failed to initialize QA system. Check cleaned_document.json and OPENAI_API_KEY.")
    else:
        print(f"✅ QA System initialized with {len(qa_system.text_chunks)} chunks.")
        app.run(debug=True, host='0.0.0.0', port=5000)