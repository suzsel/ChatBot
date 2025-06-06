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
        self.display_data = None
        self.initialize_system()

    def load_json(self, file_path):
        """Load and process JSON file for embeddings (cleaned_data.json format)"""
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
                                'title': section.get('title'),
                                'source': 'cleaned_data'
                            })
            return texts, metadata
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return [], []

    def load_clean_json_for_embeddings(self, file_path):
        """Load clean.json and convert to embedding format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            texts = []
            metadata = []

            for item in data:
                chapter_num = item.get('chapterNum')
                title_num = item.get('titleNum')

                sections = item.get('sections', [])
                for section in sections:
                    text = section.get('text', '')
                    if text:
                        # Clean HTML tags from text
                        text = re.sub(r'<[^>]+>', '', text)
                        text = re.sub(r'	', '\t', text)
                        text = text.strip()

                        if text:
                            texts.append(text)
                            metadata.append({
                                'chapterNum': chapter_num,
                                'subChapterNum': title_num or 'General',
                                'sectionNum': section.get('sectionNum', ''),
                                'title': section.get('title', ''),
                                'source': 'clean'
                            })
            return texts, metadata
        except Exception as e:
            logger.error(f"Error reading clean JSON for embeddings: {e}")
            return [], []

    def initialize_system(self):
        """Initialize the QA system with both JSON datasets"""
        try:
            all_texts = []
            all_metadata = []

            # Load embeddings data (cleaned_data.json)
            embeddings_path = 'OREC/cleaned_data.json'
            if os.path.exists(embeddings_path):
                texts1, metadata1 = self.load_json(embeddings_path)
                all_texts.extend(texts1)
                all_metadata.extend(metadata1)
                logger.info(f"Loaded {len(texts1)} texts from cleaned_data.json")

            # Load clean.json for embeddings
            clean_path = 'OREC/clean.json'
            if os.path.exists(clean_path):
                texts2, metadata2 = self.load_clean_json_for_embeddings(clean_path)
                all_texts.extend(texts2)
                all_metadata.extend(metadata2)
                logger.info(f"Loaded {len(texts2)} texts from clean.json")

                # Also load for display
                raw_display_data = self.load_display_json(clean_path)
                self.display_data = self.convert_display_data_to_hierarchy(raw_display_data)

            if not all_texts:
                logger.error("No text extracted from JSON files")
                return False

            # Split into chunks for embeddings
            char_text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            self.text_chunks = []
            self.sections_metadata = []
            for text, meta in zip(all_texts, all_metadata):
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
            self.llm = OpenAI(model_name="gpt-4.1-nano", temperature=0, max_tokens=500)
            self.chain = load_qa_chain(self.llm, chain_type="stuff")

            logger.info(f"System initialized with {len(self.text_chunks)} total text chunks from both datasets")
            return True
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False

    def get_answer(self, query, k=6):
        """Get answer for a query with source tracking from both datasets"""
        try:
            if not self.docsearch or not self.chain:
                return {
                    'answer': 'System not properly initialized.',
                    'sources': [],
                    'error': True
                }

            # Get similar documents from both datasets
            docs = self.docsearch.similarity_search(query, k=k)
            if not docs:
                return {
                    'answer': 'No relevant information found in the document.',
                    'sources': [],
                    'no_docs': True
                }

            # Enhanced prompt for better formatting
            enhanced_query = f"""
            Based on the OREC Code and Rule Book, provide a clear and well-structured answer to: {query}

            Guidelines for formatting your response:
            - Provide a direct, comprehensive yet concise answer based only on the document content
            - Use clear, natural language with proper sentence structure
            - Include specific details like timeframes, amounts, or conditions when available
            - If no information is available, state: "I don't have information about this topic"
            - Organize the response logically with smooth transitions between ideas
            - Use line breaks to separate distinct sections or categories
            - When listing requirements, fees, or steps, use numbered format (1., 2., 3.)
            - When discussing different types or categories, start each on a new line with appropriate headers
            - Bold key terms like 'license', 'applicants', 'requirements', 'fees', 'examination'
            - Use proper paragraph breaks for readability
            - Indent sub-items appropriately
            """

            with get_openai_callback() as cb:
                response = self.chain.run(input_documents=docs, question=enhanced_query)
                logger.info(f"Query cost: {cb}")

            # Check for no-info response
            no_info_indicators = ["i don't have", "not mentioned", "not specified", "not provided"]
            if any(indicator in response.lower() for indicator in no_info_indicators):
                return {
                    'answer': 'I don\'t have specific information about that topic in the OREC Code and Rule Book.',
                    'sources': [],
                    'no_answer': True
                }

            # Format the response for better readability
            formatted_response = self.format_response(response)

            # Get top 3 most relevant sections with source info
            sources = []
            for doc in docs[:3]:  # Take the top 3 documents
                source = doc.metadata if doc else None
                if source:
                    source['data_source'] = source.get('source', 'unknown')
                    sources.append(source)

            return {
                'answer': formatted_response,
                'sources': sources,  # Return a list of sources
                'error': False
            }
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            return {
                'answer': 'Error processing your question.',
                'sources': [],
                'error': True
            }

    def load_display_json(self, file_path):
        """Load JSON file for display purposes (clean.json format)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            logger.error(f"Error reading display JSON: {e}")
            return []

    def convert_display_data_to_hierarchy(self, data):
        """Convert clean.json format to hierarchical structure for display"""
        try:
            # Group by chapter
            chapters = {}

            for item in data:
                chapter_num = item.get('chapterNum')
                title_num = item.get('titleNum')

                if chapter_num not in chapters:
                    chapters[chapter_num] = {
                        'chapterNum': chapter_num,
                        'title': item.get('title', f'Chapter {chapter_num}'),
                        'subchapters': {}
                    }

                # Handle sections within the item
                sections = item.get('sections', [])
                if sections:
                    # Use titleNum as subchapter identifier
                    subchapter_key = title_num or 'General'

                    if subchapter_key not in chapters[chapter_num]['subchapters']:
                        chapters[chapter_num]['subchapters'][subchapter_key] = {
                            'subChapterNum': subchapter_key,
                            'title': item.get('title', f'Title {subchapter_key}'),
                            'sections': []
                        }

                    for section in sections:
                        # Clean HTML tags from text
                        text = section.get('text', '')
                        if text:
                            # Remove HTML tags
                            text = re.sub(r'<[^>]+>', '', text)
                            text = re.sub(r'	', '\t', text)
                            text = text.strip()

                        chapters[chapter_num]['subchapters'][subchapter_key]['sections'].append({
                            'sectionNum': section.get('sectionNum', ''),
                            'title': section.get('title', ''),
                            'text': text
                        })

            # Convert to list format expected by frontend
            result = []
            for chapter_num in sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                chapter = chapters[chapter_num]
                chapter['subchapters'] = list(chapter['subchapters'].values())
                result.append(chapter)

            return result

        except Exception as e:
            logger.error(f"Error converting display data: {e}")
            return []

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

    def format_response(self, text):
        """Format response text for better readability with improved structure and proper section headers"""
        # Clean up the text first
        text = text.strip()

        # Remove excessive whitespace but preserve intentional structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)

        # Split into sections and process each
        sections = []
        current_section = []

        # Split by double newlines to identify natural sections
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            paragraph = paragraph.strip()

            # Check if this paragraph is a main section header
            if self.is_section_header(paragraph):
                # If we have content in current section, save it
                if current_section:
                    sections.append(self.format_section(current_section))
                    current_section = []

                # Start new section with this header
                current_section = [paragraph]
            else:
                # Add to current section
                current_section.append(paragraph)

        # Don't forget the last section
        if current_section:
            sections.append(self.format_section(current_section))

        # Join all sections
        formatted_text = '\n\n'.join(sections)

        # Apply bold formatting to key terms
        formatted_text = self.apply_bold_formatting(formatted_text)

        return formatted_text

    def is_section_header(self, text):
        """Determine if a text line is a section header"""
        # Common patterns for section headers
        header_patterns = [
            r'^(Use of Registered Name|Restrictions on Advertising Content|Promotion of Incentives):?\s*$',
            r'^(Specific Restrictions for Associates and Team Advertising):?\s*$',
            r'^(Associates|Teams|Brokers?):?\s*$',
            r'^(Requirements?|Fees?|Applications?|Examinations?):?\s*$',
            r'^(For\s+\w+.*?):?\s*$',
            r'^(\w+\s+License\s*):?\s*$',
            r'^(\w+\s+Applicants?\s*):?\s*$',
            r'^[A-Z][A-Za-z\s&]+:$',  # General capitalized headers ending with colon
        ]

        for pattern in header_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Check if it's a short line that looks like a header (under 50 chars, mostly caps or title case)
        if len(text) < 50 and ':' in text:
            words = text.replace(':', '').split()
            if len(words) <= 6 and all(word[0].isupper() for word in words if word):
                return True

        return False

    def format_section(self, section_lines):
        """Format a single section with proper header and content structure"""
        if not section_lines:
            return ""

        formatted_lines = []
        header = section_lines[0]
        content_lines = section_lines[1:] if len(section_lines) > 1 else []

        # Format the header
        if self.is_section_header(header):
            # Clean up header formatting
            header = header.rstrip(':').strip()
            formatted_lines.append(f'<div class="section-header"><h3>{header}</h3></div>')
        else:
            # If first line isn't a header, treat it as regular content
            content_lines.insert(0, header)

        # Process content lines
        if content_lines:
            formatted_content = self.format_content_lines(content_lines)
            formatted_lines.append(formatted_content)

        return '\n'.join(formatted_lines)

    def format_content_lines(self, lines):
        """Format content lines with proper paragraph and list structure"""
        formatted_parts = []
        current_list = []
        list_type = None  # 'ordered' or 'unordered'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for numbered list items
            if re.match(r'^\d+\.\s+', line):
                # End any previous list of different type
                if current_list and list_type != 'ordered':
                    formatted_parts.append(self.close_list(current_list, list_type))
                    current_list = []

                # Start or continue ordered list
                list_type = 'ordered'
                item_text = re.sub(r'^\d+\.\s+', '', line)
                current_list.append(item_text)
                continue

            # Check for bullet points
            if re.match(r'^[-•*]\s+', line):
                # End any previous list of different type
                if current_list and list_type != 'unordered':
                    formatted_parts.append(self.close_list(current_list, list_type))
                    current_list = []

                # Start or continue unordered list
                list_type = 'unordered'
                item_text = re.sub(r'^[-•*]\s+', '', line)
                current_list.append(item_text)
                continue

            # Regular paragraph - close any open list first
            if current_list:
                formatted_parts.append(self.close_list(current_list, list_type))
                current_list = []
                list_type = None

            # Add as regular paragraph
            formatted_parts.append(f'<p>{line}</p>')

        # Close any remaining list
        if current_list:
            formatted_parts.append(self.close_list(current_list, list_type))

        return '\n'.join(formatted_parts)

    def close_list(self, items, list_type):
        """Close a list with proper HTML tags"""
        if not items:
            return ""

        list_items = '\n'.join([f'<li>{item}</li>' for item in items])

        if list_type == 'ordered':
            return f'<ol class="formatted-list">\n{list_items}\n</ol>'
        else:
            return f'<ul class="formatted-list">\n{list_items}\n</ul>'

    def apply_bold_formatting(self, text):
        """Apply bold formatting to key terms"""
        # Convert **text** to <strong>text</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

        # Bold key terms if not already formatted
        key_terms = [
            'license', 'applicants', 'requirements', 'fees', 'examination',
            'application', 'broker', 'associate', 'advertising', 'registered',
            'business', 'trade name', 'franchise', 'permission', 'property',
            'owner', 'representative', 'social networking', 'yard signs',
            'incentives', 'seller', 'supervision', 'teams', 'approved'
        ]

        for term in key_terms:
            # Only bold if not already in HTML tags
            pattern = rf'\b(?<!<[^>]*>)({re.escape(term)})(?![^<]*>)\b'
            replacement = r'<strong>\1</strong>'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def get_orec_sections(self):
        """Get structured OREC Code & Rules Book sections from cleaned_data.json"""
        try:
            # Load cleaned_data.json directly for display
            with open('OREC/cleaned_data.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            logger.error(f"Error getting OREC sections: {e}")
            return []

    def get_document_sections_from_embeddings(self):
        """Get structured document sections from embeddings data (fallback)"""
        try:
            # This is now specifically for cleaned_data.json
            return self.get_orec_sections()
        except Exception as e:
            logger.error(f"Error getting sections from embeddings: {e}")
            return []

    def get_document_sections(self):
        """Get structured Statutes sections from clean.json for frontend display"""
        try:
            if self.display_data:
                return self.display_data
            else:
                # Fallback to empty if no display data
                logger.warning("No display data available for Statutes")
                return []
        except Exception as e:
            logger.error(f"Error getting Statutes sections: {e}")
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
        'chunks_loaded': len(qa_system.text_chunks),
        'statutes_data_loaded': qa_system.display_data is not None,
        'orec_data_loaded': qa_system.get_orec_sections() is not None,
        'statutes_chapters': len(qa_system.display_data) if qa_system.display_data else 0,
        'orec_chapters': len(qa_system.get_orec_sections()) if qa_system.get_orec_sections() else 0
    })


@app.route('/api/statutes-sections', methods=['GET'])
def get_statutes_sections():
    """Get Statutes document sections (clean.json) for frontend display"""
    try:
        sections = qa_system.get_document_sections()  # This returns clean.json data
        return jsonify({'sections': sections, 'total': len(sections)})
    except Exception as e:
        logger.error(f"Error getting statutes sections: {e}")
        return jsonify({'error': 'Failed to load statutes document sections'}), 500


@app.route('/api/orec-sections', methods=['GET'])
def get_orec_sections():
    """Get OREC Code & Rules Book sections (cleaned_data.json) for frontend display"""
    try:
        sections = qa_system.get_orec_sections()
        return jsonify({'sections': sections, 'total': len(sections)})
    except Exception as e:
        logger.error(f"Error getting OREC sections: {e}")
        return jsonify({'error': 'Failed to load OREC document sections'}), 500


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
            'sources': result['sources'],
            'type': 'success' if not result.get('error') and not result.get('no_answer') else 'no_answer' if result.get(
                'no_answer') else 'error'
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': 'Internal server error', 'type': 'error'}), 500


if __name__ == '__main__':
    if qa_system.docsearch is None:
        print("❌ Failed to initialize QA system. Check cleaned_data.json and OPENAI_API_KEY.")
    else:
        print(f"✅ QA System initialized with {len(qa_system.text_chunks)} chunks.")
        if qa_system.display_data:
            print(f"✅ Display data loaded with {len(qa_system.display_data)} chapters.")
        else:
            print("⚠️ Display data not loaded, using embeddings data for display.")
        app.run(debug=True, host='0.0.0.0', port=5000)