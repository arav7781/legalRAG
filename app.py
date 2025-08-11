import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import quote_plus, urlencode
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from serpapi import GoogleSearch
from firecrawl import FirecrawlApp
from google import genai
import sqlite3

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize external services
f_app = FirecrawlApp(api_key="")
gemini_client = genai.Client(api_key="")

# API Keys and configuration
INDIAN_KANOON_API_KEY = ""
SERPAPI_API_KEY = ""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are an AI legal research assistant specialized in Indian law.")

# Database initialization
def init_db():
    """Initialize SQLite database for caching search results"""
    conn = sqlite3.connect('legal_cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            source TEXT,
            results TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Legal APIs and Data Sources
class IndianLegalAPIs:
    """Class to handle various Indian legal data sources"""
    
    def __init__(self):
        self.indian_kanoon_base = "https://api.indiankanoon.org"
        self.ecourts_base = "https://judgments.ecourts.gov.in"
        self.sci_base = "https://www.sci.gov.in"
        
    def search_indian_kanoon(self, query: str, doctypes: str = None, 
                           fromdate: str = None, todate: str = None, 
                           maxcites: int = 10) -> Dict:
        """Search Indian Kanoon database - Fixed to use correct method"""
        # Build the URL with query parameters
        base_url = f"{self.indian_kanoon_base}/search/"
        params = {
            'formInput': query,
            'pagenum': 0,
            'maxcites': maxcites
        }
        
        if doctypes:
            params['doctypes'] = doctypes
        if fromdate:
            params['fromdate'] = fromdate
        if todate:
            params['todate'] = todate
            
        # Convert params to query string
        query_string = urlencode(params)
        full_url = f"{base_url}?{query_string}"
        
        headers = {
            'Authorization': f'Token {INDIAN_KANOON_API_KEY}',
            'Accept': 'application/json',
            'User-Agent': 'Legal Research App/1.0'
        }
        
        try:
            # Use POST method instead of GET as per Indian Kanoon API docs
            response = requests.post(full_url, headers=headers, timeout=30)
            print(f"Indian Kanoon API Response Status: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # Sometimes the API returns HTML, try to parse it
                    return self._parse_html_response(response.text, query)
            elif response.status_code == 403:
                return {"error": "API authentication failed - check your token"}
            else:
                return {"error": f"API returned status code {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            print(f"Indian Kanoon API Error: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}
    
    def _parse_html_response(self, html_content: str, query: str) -> Dict:
        """Parse HTML response when JSON parsing fails"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for search results in the HTML
            results = []
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:10]:  # Limit to 10 results
                title_elem = div.find('a')
                snippet_elem = div.find('div', class_='snippet')
                
                if title_elem:
                    result = {
                        'title': title_elem.get_text(strip=True),
                        'link': title_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'source': 'indiankanoon.org'
                    }
                    results.append(result)
            
            return {
                'docs': results,
                'found': len(results),
                'query': query
            }
        except Exception as e:
            return {"error": f"Failed to parse HTML response: {str(e)}"}
    
    def search_alternative_method(self, query: str) -> Dict:
        """Alternative search method using direct web scraping"""
        try:
            search_url = f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"
            scrape_result = f_app.scrape_url(search_url, formats=['markdown', 'html'])
            
            # Parse the scraped content
            return self._parse_indiankanoon_scrape(scrape_result.html, query)
        except Exception as e:
            return {"error": f"Alternative search failed: {str(e)}"}
    
    def _parse_indiankanoon_scrape(self, html: str, query: str) -> Dict:
        """Parse scraped Indian Kanoon search results"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            results = []
            # Look for various possible result containers
            result_selectors = [
                '.result',
                '.search_result',
                'div[class*="result"]',
                'a[href*="/doc/"]'
            ]
            
            for selector in result_selectors:
                elements = soup.select(selector)
                if elements:
                    break
            
            for elem in elements[:10]:
                title = elem.get_text(strip=True)
                link = elem.get('href', '') if elem.name == 'a' else elem.find('a', href=True)
                
                if title and len(title) > 10:  # Filter out very short titles
                    result = {
                        'title': title,
                        'link': link['href'] if isinstance(link, dict) else (link or ''),
                        'snippet': title[:200] + '...' if len(title) > 200 else title,
                        'source': 'indiankanoon.org'
                    }
                    results.append(result)
            
            return {
                'docs': results,
                'found': len(results),
                'query': query
            }
        except Exception as e:
            return {"error": f"Failed to parse scraped content: {str(e)}"}
    
    def get_document(self, doc_id: str, maxcites: int = 10, maxcitedby: int = 10) -> Dict:
        """Get full document from Indian Kanoon"""
        url = f"{self.indian_kanoon_base}/doc/{doc_id}/"
        params = {
            'maxcites': maxcites,
            'maxcitedby': maxcitedby
        }
        
        headers = {
            'Authorization': f'Token {INDIAN_KANOON_API_KEY}',
            'Accept': 'application/json',
            'User-Agent': 'Legal Research App/1.0'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

class SupremeCourtAPI:
    """Handle Supreme Court of India data"""
    
    def __init__(self):
        self.base_url = "https://www.sci.gov.in"
        self.judgments_url = "https://main.sci.gov.in/judgments"
    
    def search_judgments(self, query: str, year: str = None) -> List[Dict]:
        """Search Supreme Court judgments using multiple methods"""
        try:
            # Method 1: Try direct search on SC website
            results = self._search_sc_direct(query, year)
            if results:
                return results
            
            # Method 2: Use SerpAPI for site-specific search
            return self._search_sc_via_google(query, year)
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def _search_sc_direct(self, query: str, year: str) -> List[Dict]:
        """Direct search on Supreme Court website"""
        try:
            search_url = f"{self.judgments_url}?search={quote_plus(query)}"
            if year:
                search_url += f"&year={year}"
                
            scrape_result = f_app.scrape_url(search_url, formats=['markdown', 'html'])
            return self._parse_sc_judgments(scrape_result.html, query, year)
        except Exception as e:
            print(f"SC direct search failed: {e}")
            return []
    
    def _search_sc_via_google(self, query: str, year: str) -> List[Dict]:
        """Search SC judgments via Google site search"""
        try:
            search_query = f"{query} site:sci.gov.in"
            if year:
                search_query += f" {year}"
            
            search = GoogleSearch({
                "api_key": SERPAPI_API_KEY,
                "engine": "google",
                "q": search_query,
                "num": 10
            })
            
            results = search.get_dict()
            judgments = []
            
            if 'organic_results' in results:
                for result in results['organic_results']:
                    if 'sci.gov.in' in result.get('link', ''):
                        judgment = {
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', ''),
                            'court': 'Supreme Court of India',
                            'source': 'sci.gov.in',
                            'relevance': 'High'
                        }
                        judgments.append(judgment)
            
            return judgments
        except Exception as e:
            print(f"SC Google search failed: {e}")
            return []
    
    def _parse_sc_judgments(self, html: str, query: str, year: str) -> List[Dict]:
        """Parse Supreme Court HTML for judgment information"""
        judgments = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for judgment links and titles
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Filter for judgment-related links
                if any(keyword in text.lower() for keyword in ['judgment', 'order', 'case', query.lower()]):
                    judgment = {
                        'title': text,
                        'link': href if href.startswith('http') else f"https://www.sci.gov.in{href}",
                        'court': 'Supreme Court of India',
                        'year': year or 'Various',
                        'source': 'sci.gov.in',
                        'relevance': 'High'
                    }
                    judgments.append(judgment)
                    
                    if len(judgments) >= 10:  # Limit results
                        break
        except Exception as e:
            print(f"SC parsing error: {e}")
            
        return judgments

class ECourtAPI:
    """Handle eCourts platform data - Enhanced version"""
    
    def __init__(self):
        self.base_url = "https://judgments.ecourts.gov.in"
        self.search_url = f"{self.base_url}/pdfsearch"
    
    def search_judgments(self, query: str, court_type: str = None, 
                        from_date: str = None, to_date: str = None) -> List[Dict]:
        """Search eCourts judgment database with improved methods"""
        try:
            # Method 1: Try direct eCourts search
            results = self._search_ecourts_direct(query, court_type, from_date, to_date)
            if results:
                return results
            
            # Method 2: Use Google site search as fallback
            return self._search_ecourts_via_google(query)
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def _search_ecourts_direct(self, query: str, court_type: str, 
                              from_date: str, to_date: str) -> List[Dict]:
        """Direct search on eCourts platform"""
        try:
            # Build search URL with parameters
            params = {
                'free_text': query,
                'from_date': from_date or '',
                'to_date': to_date or '',
                'court_type': court_type or ''
            }
            
            # Remove empty parameters
            params = {k: v for k, v in params.items() if v}
            
            if params:
                search_url = f"{self.search_url}?{urlencode(params)}"
            else:
                search_url = self.search_url
                
            scrape_result = f_app.scrape_url(search_url, formats=['markdown', 'html'])
            return self._parse_ecourt_results(scrape_result.html, query)
        except Exception as e:
            print(f"eCourts direct search failed: {e}")
            return []
    
    def _search_ecourts_via_google(self, query: str) -> List[Dict]:
        """Search eCourts via Google site search"""
        try:
            search_query = f"{query} site:ecourts.gov.in OR site:judgments.ecourts.gov.in"
            
            search = GoogleSearch({
                "api_key": SERPAPI_API_KEY,
                "engine": "google",
                "q": search_query,
                "num": 10
            })
            
            results = search.get_dict()
            judgments = []
            
            if 'organic_results' in results:
                for result in results['organic_results']:
                    if 'ecourts.gov.in' in result.get('link', ''):
                        judgment = {
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', ''),
                            'court': 'High Court/District Court',
                            'source': 'ecourts.gov.in',
                            'relevance': 'Medium'
                        }
                        judgments.append(judgment)
            
            return judgments
        except Exception as e:
            print(f"eCourts Google search failed: {e}")
            return []
    
    def _parse_ecourt_results(self, html: str, query: str) -> List[Dict]:
        """Parse eCourts search results"""
        judgments = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for various judgment-related elements
            selectors = [
                'a[href*=".pdf"]',  # PDF links
                '.judgment-link',
                'a[href*="judgment"]',
                'a[href*="order"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for elem in elements:
                    title = elem.get_text(strip=True)
                    link = elem.get('href', '')
                    
                    if title and len(title) > 5:
                        judgment = {
                            'title': title,
                            'link': link if link.startswith('http') else f"{self.base_url}{link}",
                            'court': 'High Court/District Court',
                            'source': 'ecourts.gov.in',
                            'relevance': 'Medium'
                        }
                        judgments.append(judgment)
                        
                        if len(judgments) >= 10:
                            break
                
                if judgments:
                    break
            
        except Exception as e:
            print(f"eCourts parsing error: {e}")
            
        return judgments

class LegalNewsAPI:
    """Enhanced legal news sources handler"""
    
    def __init__(self):
        self.sources = {
            'barandbench': 'https://www.barandbench.com',
            'livelaw': 'https://www.livelaw.in',
            'scobserver': 'https://www.scobserver.in',
            'thelogicalindian': 'https://thelogicalindian.com'
        }
    
    def get_legal_news(self, query: str, days: int = 30) -> List[Dict]:
        """Get recent legal news related to query with improved search"""
        news_items = []
        
        # Generic legal news search
        try:
            general_search = GoogleSearch({
                "api_key": SERPAPI_API_KEY,
                "engine": "google_news",
                "q": f"{query} legal news India",
                "gl": "in",
                "hl": "en",
                "num": 20
            })
            
            results = general_search.get_dict()
            if 'news_results' in results:
                for result in results['news_results'][:10]:
                    news_items.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': result.get('source', {}).get('name', 'General News'),
                        'date': result.get('date', ''),
                        'thumbnail': result.get('thumbnail', '')
                    })
        except Exception as e:
            print(f"Error fetching general legal news: {e}")
        
        # Source-specific searches
        for source, base_url in self.sources.items():
            try:
                search = GoogleSearch({
                    "api_key": SERPAPI_API_KEY,
                    "engine": "google",
                    "q": f"{query} site:{base_url}",
                    "tbs": f"qdr:d{days}",
                    "num": 5
                })
                
                results = search.get_dict()
                if 'organic_results' in results:
                    for result in results['organic_results']:
                        news_items.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', ''),
                            'source': source,
                            'date': result.get('date', '')
                        })
            except Exception as e:
                print(f"Error fetching news from {source}: {e}")
                
        return news_items

# Main Legal Research Engine - Enhanced
class LegalResearchEngine:
    """Main class that orchestrates all legal research operations"""
    
    def __init__(self):
        self.ik_api = IndianLegalAPIs()
        self.sc_api = SupremeCourtAPI()
        self.ecourt_api = ECourtAPI()
        self.news_api = LegalNewsAPI()
    
    def comprehensive_search(self, query: str, search_type: str = "all", 
                           filters: Dict = None) -> Dict:
        """Perform comprehensive legal search across all sources"""
        results = {
            'query': query,
            'search_type': search_type,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        filters = filters or {}
        
        # Indian Kanoon Search with fallback
        if search_type in ['all', 'cases', 'judgments']:
            print("Searching Indian Kanoon...")
            try:
                # Try API first
                ik_results = self.ik_api.search_indian_kanoon(
                    query,
                    doctypes=filters.get('doctypes'),
                    fromdate=filters.get('fromdate'),
                    todate=filters.get('todate')
                )
                
                # If API fails, try alternative method
                if 'error' in ik_results:
                    print("API failed, trying alternative method...")
                    ik_results = self.ik_api.search_alternative_method(query)
                
                results['sources']['indian_kanoon'] = ik_results
            except Exception as e:
                results['sources']['indian_kanoon'] = {"error": f"Search failed: {str(e)}"}
        
        # Supreme Court Search
        if search_type in ['all', 'supreme_court']:
            print("Searching Supreme Court...")
            try:
                sc_results = self.sc_api.search_judgments(query, filters.get('year'))
                results['sources']['supreme_court'] = sc_results
            except Exception as e:
                results['sources']['supreme_court'] = [{"error": f"Search failed: {str(e)}"}]
        
        # eCourts Search
        if search_type in ['all', 'high_courts', 'district_courts']:
            print("Searching eCourts...")
            try:
                ecourt_results = self.ecourt_api.search_judgments(
                    query,
                    court_type=filters.get('court_type'),
                    from_date=filters.get('fromdate'),
                    to_date=filters.get('todate')
                )
                results['sources']['ecourts'] = ecourt_results
            except Exception as e:
                results['sources']['ecourts'] = [{"error": f"Search failed: {str(e)}"}]
        
        # Legal News
        if search_type in ['all', 'news']:
            print("Searching Legal News...")
            try:
                news_results = self.news_api.get_legal_news(query, filters.get('days', 30))
                results['sources']['legal_news'] = news_results
            except Exception as e:
                results['sources']['legal_news'] = [{"error": f"Search failed: {str(e)}"}]
        
        return results
    
    def analyze_with_ai(self, search_results: Dict, analysis_type: str = "summary") -> Dict:
        """Use AI to analyze and summarize search results"""
        try:
            chat_client = gemini_client.chats.create(model="gemini-2.5-flash")
            
            prompt = f"""
            As a legal research assistant specializing in Indian law, analyze the following search results for the query: "{search_results['query']}"
            
            Search Results: {json.dumps(search_results, indent=2)}
            
            Please provide a comprehensive analysis including:
            1. **Executive Summary**: Key findings and overall assessment
            2. **Relevant Cases**: Most important judgments found with brief descriptions
            3. **Legal Precedents**: Important legal principles established
            4. **Recent Developments**: Any recent trends or changes in law
            5. **Statutory Framework**: Relevant laws, sections, and regulations
            6. **Research Recommendations**: Areas for further investigation
            7. **Practical Implications**: How this affects legal practice
            
            Format the response in a clear, structured manner suitable for legal professionals.
            """
            
            response = chat_client.send_message(prompt)
            analysis = response.candidates[0].content.parts[0].text
            
            return {
                'analysis_type': analysis_type,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f"AI analysis failed: {str(e)}"}

# Initialize research engine
research_engine = LegalResearchEngine()

# API Routes
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "message": "Indian Legal Research API - Ready",
        "version": "2.0 - Enhanced",
        "timestamp": datetime.now().isoformat(),
        "status": "operational"
    })

@app.route("/search", methods=["POST"])
def comprehensive_search():
    """Main search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        search_type = data.get('search_type', 'all')
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"Processing search: {query}")
        results = research_engine.comprehensive_search(query, search_type, filters)
        return jsonify(results)
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/analyze", methods=["POST"])
def ai_analysis():
    """AI analysis endpoint"""
    try:
        data = request.get_json()
        search_results = data.get('search_results', {})
        analysis_type = data.get('analysis_type', 'comprehensive')
        
        if not search_results:
            return jsonify({'error': 'Search results are required'}), 400
        
        print("Starting AI analysis...")
        analysis = research_engine.analyze_with_ai(search_results, analysis_type)
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/document/<doc_id>", methods=["GET"])
def get_document(doc_id):
    """Get full document by ID"""
    try:
        maxcites = request.args.get('maxcites', 10, type=int)
        maxcitedby = request.args.get('maxcitedby', 10, type=int)
        
        document = research_engine.ik_api.get_document(doc_id, maxcites, maxcitedby)
        return jsonify(document)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/news", methods=["GET"])
def legal_news():
    """Get legal news"""
    try:
        query = request.args.get('query', 'legal news India')
        days = request.args.get('days', 7, type=int)
        
        news = research_engine.news_api.get_legal_news(query, days)
        return jsonify({'news': news, 'count': len(news)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/courts", methods=["GET"])
def get_court_types():
    """Get available court types for filtering"""
    court_types = {
        'supreme_court': 'Supreme Court of India',
        'high_courts': 'High Courts',
        'district_courts': 'District Courts',
        'tribunals': 'Tribunals',
        'consumer_forums': 'Consumer Forums'
    }
    
    doctypes = {
        'supremecourt': 'Supreme Court',
        'delhi': 'Delhi High Court',
        'bombay': 'Bombay High Court',
        'kolkata': 'Kolkata High Court',
        'chennai': 'Madras High Court',
        'allahabad': 'Allahabad High Court',
        'andhra': 'Andhra Pradesh High Court',
        'gujarat': 'Gujarat High Court',
        'karnataka': 'Karnataka High Court',
        'kerala': 'Kerala High Court',
        'madhyapradesh': 'Madhya Pradesh High Court',
        'orissa': 'Orissa High Court',
        'punjab': 'Punjab & Haryana High Court',
        'rajasthan': 'Rajasthan High Court',
        'tribunals': 'All Tribunals',
        'delhidc': 'Delhi District Courts'
    }
    
    return jsonify({
        'court_types': court_types,
        'indian_kanoon_doctypes': doctypes
    })

@app.route("/test", methods=["GET"])
def test_apis():
    """Test endpoint to check API connectivity"""
    test_results = {}
    
    # Test Indian Kanoon API
    try:
        ik_test = research_engine.ik_api.search_indian_kanoon("test", maxcites=1)
        test_results['indian_kanoon'] = "Success" if 'error' not in ik_test else f"Failed: {ik_test['error']}"
    except Exception as e:
        test_results['indian_kanoon'] = f"Error: {str(e)}"
    
    # Test SerpAPI
    try:
        search = GoogleSearch({
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "q": "test",
            "num": 1
        })
        results = search.get_dict()
        test_results['serpapi'] = "Success" if 'organic_results' in results else "Failed"
    except Exception as e:
        test_results['serpapi'] = f"Error: {str(e)}"
    
    # Test Firecrawl
    try:
        test_scrape = f_app.scrape_url("https://example.com", formats=['html'])
        test_results['firecrawl'] = "Success" if test_scrape else "Failed"
    except Exception as e:
        test_results['firecrawl'] = f"Error: {str(e)}"
    
    return jsonify(test_results)

if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    print("Starting Indian Legal Research API...")
    app.run(debug=True, host='0.0.0.0', port=5000)