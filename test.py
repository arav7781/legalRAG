import requests
import json
import time
import asyncio
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemTester:
    def __init__(self, base_url: str = "http://localhost:8000", bearer_token: str = None):
        self.base_url = base_url
        self.bearer_token = bearer_token or "legal_doc_analyzer_token_2024"
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Use the Vinay Sharma legal case PDF here
        self.document_url = "https://api.sci.gov.in/supremecourt/2020/5529/5529_2020_5_301_20686_Judgement_14-Feb-2020.pdf"

    def test_health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: Status {response.status_code}")
                logger.error(f"Health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            logger.error(f"Health check error: {str(e)}")
            return False

    def test_knowledge_base_stats(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/knowledge-base/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Knowledge base stats: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"⚠️  Knowledge base stats endpoint not available: Status {response.status_code}")
                return True  # Not critical
        except Exception as e:
            print(f"⚠️  Knowledge base stats error: {str(e)}")
            return True  # Not critical

    def test_sample_query(self, questions_limit: int = None) -> bool:
        # Example legal questions related to the Vinay Sharma / Nirbhaya case
        sample_questions = [
            "What was the Supreme Court's decision on Vinay Sharma's mercy petition?",
            "What legal grounds were raised by Vinay Sharma in his mercy petition?",
            "What is the significance of the 'rarest of rare' doctrine in this case?",
            # "What were the key reasons for rejecting the mercy petition?",
            # "How did the court address allegations of torture and mental illness?",
            # "What is the final sentence awarded to Vinay Sharma?",
            # "What role did the President of India play in this case?",
            # "What sections of the IPC were invoked in the conviction?",
            # "How did the court consider the victim's family during sentencing?",
            # "What precedent does this case set for future mercy petitions?"
        ]

        if questions_limit:
            sample_questions = sample_questions[:questions_limit]

        sample_data = {
            "documents": self.document_url,
            "questions": sample_questions
        }

        try:
            print(f"🔄 Testing {len(sample_questions)} legal questions on Vinay Sharma case...")
            print(f"📄 Document URL: {self.document_url}")
            print("🔧 System will download and process the legal document in real-time")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/legal/analyze",
                headers=self.headers,
                json=sample_data,
                timeout=120
            )

            end_time = time.time()
            latency = end_time - start_time

            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])

                print(f"✅ Query successful (Latency: {latency:.2f}s)")
                print(f"📊 Received {len(answers)} answers")
                
                print("\n" + "="*100)
                print("🔍 DETAILED ANSWERS:")
                print("="*100)
                
                for i, (question, answer) in enumerate(zip(sample_questions, answers)):
                    print(f"\n{'='*20} QUESTION {i+1} {'='*20}")
                    print(f"❓ {question}")
                    print(f"\n💡 COMPLETE ANSWER:")
                    print("-" * 60)
                    print(answer)
                    print("-" * 60)
                    print(f"📊 Answer length: {len(answer)} characters")
                    
                    if len(answer.strip()) < 50:
                        print("⚠️  WARNING: Very short answer")
                    elif "Information not available" in answer:
                        print("ℹ️  INFO: No relevant information found")
                    elif "Error processing" in answer:
                        print("❌ ERROR: Processing error detected")
                    else:
                        print("✅ GOOD: Substantial answer provided")

                print(f"\n{'='*50}")
                print("📊 SUMMARY STATISTICS:")
                print(f"{'='*50}")
                
                total_answers = len(answers)
                meaningful_answers = [a for a in answers if len(a.strip()) > 50 and "Information not available" not in a and "Error processing" not in a]
                error_answers = [a for a in answers if "Error processing" in a]
                no_info_answers = [a for a in answers if "Information not available" in a]
                
                print(f"Total Questions: {len(sample_questions)}")
                print(f"Total Answers: {total_answers}")
                print(f"Meaningful Answers: {len(meaningful_answers)} ({len(meaningful_answers)/total_answers*100:.1f}%)")
                print(f"No Information Found: {len(no_info_answers)} ({len(no_info_answers)/total_answers*100:.1f}%)")
                print(f"Processing Errors: {len(error_answers)} ({len(error_answers)/total_answers*100:.1f}%)")
                print(f"Average Answer Length: {sum(len(a) for a in answers)/len(answers):.0f} characters")
                
                success_rate = len(meaningful_answers) / total_answers * 100
                if success_rate >= 70:
                    print(f"🎉 EXCELLENT: {success_rate:.1f}% meaningful answers")
                elif success_rate >= 50:
                    print(f"✅ GOOD: {success_rate:.1f}% meaningful answers")
                elif success_rate >= 30:
                    print(f"⚠️  FAIR: {success_rate:.1f}% meaningful answers")
                else:
                    print(f"❌ POOR: {success_rate:.1f}% meaningful answers")

                return len(answers) == len(sample_questions) and success_rate >= 30

            else:
                print(f"❌ Query failed: Status {response.status_code}")
                print(f"Response: {response.text}")
                logger.error(f"Query failed: Status {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            logger.error(f"Query error: {str(e)}")
            return False

    def test_single_question(self, question: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/legal/analyze",
                headers=self.headers,
                json={
                    "documents": self.document_url,
                    "questions": [question]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])
                return answers[0] if answers else "No answer received"
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    def interactive_test(self):
        print("\n🎮 INTERACTIVE TESTING MODE")
        print("="*50)
        print("Type your questions (or 'quit' to exit):")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if not question:
                continue
                
            print(f"\n🔄 Processing: {question}")
            print("-" * 60)
            
            start_time = time.time()
            answer = self.test_single_question(question)
            end_time = time.time()
            
            print(f"💡 ANSWER (took {end_time-start_time:.2f}s):")
            print(answer)
            print("-" * 60)

    async def run_comprehensive_test(self, questions_limit: int = None):
        print("🚀 COMPREHENSIVE RAG SYSTEM TEST")
        print("="*80)
        print(f"🎯 Target System: {self.base_url}")
        print("="*80)

        results = {}
        
        print("\n1️⃣  Testing Health Check...")
        results["health"] = self.test_health_check()
        
        print("\n2️⃣  Testing Knowledge Base...")
        results["knowledge_base"] = self.test_knowledge_base_stats()
        
        print("\n3️⃣  Testing Sample Queries...")
        results["sample_queries"] = self.test_sample_query(questions_limit)
        
        print(f"\n{'='*80}")
        print("🏁 FINAL RESULTS:")
        print(f"{'='*80}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title(): <20} {status}")
        
        success_rate = (passed / total) * 100
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
        
        if success_rate == 100:
            print("🏆 PERFECT! All tests passed!")
        elif success_rate >= 80:
            print("🎉 EXCELLENT! System performing very well!")
        elif success_rate >= 60:
            print("✅ GOOD! System working with minor issues!")
        else:
            print("⚠️  NEEDS ATTENTION! Multiple issues detected!")
        
        return success_rate >= 60

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test RAG System - Legal Doc (Vinay Sharma Case)')
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='Base URL of the RAG system')
    parser.add_argument('--token', help='Bearer token for authentication')
    parser.add_argument('--questions', type=int, 
                       help='Limit number of test questions (default: all)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    tester = RAGSystemTester(base_url=args.base_url, bearer_token=args.token)
    
    if args.interactive:
        tester.interactive_test()
    else:
        asyncio.run(tester.run_comprehensive_test(args.questions))

if __name__ == "__main__":
    main()
