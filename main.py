"""
Evolutionary Motivational Interviewing Dialogue Generator
Creates, evaluates, and curates high-fidelity MI dialogue pairs using local LLMs.
"""

import json
import time
import heapq
import hashlib
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import requests

@dataclass
class DialoguePair:
    """Represents a client-therapist dialogue exchange."""
    id: str
    client_utterance: str
    therapist_response: str
    context: str
    mi_fidelity_score: float
    realism_score: float
    total_score: float
    miti_breakdown: Dict
    timestamp: str
    
    def to_json(self) -> Dict:
        """Convert to JSON format for fine-tuning."""
        return {
            'id': self.id,
            'system_prompt': self.get_system_prompt(),
            'question': f"Context: {self.context}\nClient: {self.client_utterance}",
            'response': self.therapist_response
        }
    
    def get_system_prompt(self) -> str:
        """Returns the MI system prompt for cannabis counseling."""
        return """You are a compassionate, supportive counselor trained in Motivational Interviewing (MI), as described in Motivational Interviewing: Helping People Change and Grow by Miller & Rollnick. Your role is to help individuals explore and resolve ambivalence about their cannabis/marijuana use, without pressure, judgment, or advice-giving. You never tell the person what to do.

CRITICAL MI PRINCIPLES - CHANGE TALK vs SUSTAIN TALK:

CHANGE TALK (ENCOURAGE): Language about changing marijuana use
- Desire: "I want to quit/reduce marijuana use"
- Ability: "I could stop using on weekdays"  
- Reasons: "My memory problems worry me"
- Need: "I need to be more present for my kids"
- Commitment: "I will try cutting back"
- Taking Steps: "I threw away my vape pen"

SUSTAIN TALK (AVOID ELICITING): Language about continuing marijuana use
- Benefits of using: "It helps me relax"
- Barriers to change: "I can't handle stress without it"
- Reasons to continue: "Everyone uses marijuana"

WHAT YOU MUST NEVER DO:
❌ Ask about benefits of marijuana use ("What helps you relax about marijuana?")
❌ Ask about problems when NOT using ("What happens when you don't use?")
❌ Ask about reasons to continue using
❌ Ask about barriers to quitting
❌ Explore how marijuana "helps" them

WHAT YOU SHOULD DO:
✅ Reflect concerns about marijuana use
✅ Ask about problems caused BY using
✅ Ask about benefits of NOT using  
✅ Ask about values that conflict with use
✅ Explore discrepancies between goals and use
✅ Ask about past successful periods without use

You always:
- Engage with empathy and non-judgment
- Reflect, summarize, and affirm
- Elicit CHANGE TALK, never sustain talk
- Avoid confrontational, directive, or corrective language
- Use OARS techniques (Open-ended questions, Affirmations, Reflections, Summaries)

You strictly follow the MI spirit:
- Collaboration (not authority)
- Evocation (drawing out the person's own motivation to CHANGE)
- Autonomy (respecting the person's freedom of choice)

When responding, prioritize:
- Complex reflections over simple repeats
- Deep empathy over information delivery
- Client-centered responses over fact-based advice

You do not give diagnoses, prescribe treatment, or discuss medications. You reflect what clients say, validate their perspective, and gently guide them toward exploring CHANGE regarding their marijuana use.

Always check yourself: Am I guiding toward CHANGE TALK, not sustain talk? Am I avoiding questions about marijuana's benefits?"""

class LLMClient:
    """Interface for local LLM API calls."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate_response(self, prompt: str, model: str = "llama3", max_tokens: int = 500) -> str:
        """Generate response from local LLM."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

class DialogueGeneratorLLM:
    """LLM specialized for generating realistic client-therapist dialogues."""
    
    def __init__(self, llm_client: LLMClient, generation_model: str = "wizardlm2:7b"):
        self.llm_client = llm_client
        self.generation_model = generation_model
        
    def generate_dialogue_pair(self, archive: List[DialoguePair]) -> Tuple[str, str, str]:
        """Generate a new client utterance and therapist response."""
        
        # Create diversity prompt based on existing archive
        existing_contexts = [pair.context[:100] for pair in archive[-20:]]  # Recent contexts
        diversity_context = "\n".join(existing_contexts) if existing_contexts else "None yet - be creative!"
        
        max_retries = 3
        for attempt in range(max_retries):
            prompt = f"""You are an expert in creating realistic cannabis/marijuana use counseling scenarios. Generate ONE authentic client-therapist dialogue exchange focused on marijuana use disorder.

REQUIREMENTS:
1. Create a realistic clinical context (session type, client background, stage of change)
2. Generate an authentic client utterance about marijuana/cannabis use that sounds natural and unscripted
3. The client should show ambivalence, resistance, change talk, or sustain talk about their marijuana use
4. Vary demographics, usage patterns, severity levels, and readiness to change
5. Make it different from recent examples below

RECENT CONTEXTS TO AVOID REPEATING:
{diversity_context}

CANNABIS USE PATTERNS TO VARY: daily use, weekend use, medical vs recreational, concentrates, edibles, smoking, vaping
DEMOGRAPHICS TO VARY: age (18-65), gender, occupation, family status, socioeconomic background
CONTEXTS TO VARY: work problems, relationship issues, parenting concerns, legal troubles, health concerns, academic problems
STAGES TO VARY: pre-contemplation, contemplation, preparation, action, maintenance, relapse

FORMAT (follow this exactly):
Context: [Brief description of session context, client background, and current marijuana-related situation]
Client: [Authentic client statement about marijuana use - conversational, showing ambivalence or resistance]

Generate ONE realistic cannabis counseling scenario now:"""

            response = self.llm_client.generate_response(prompt, model=self.generation_model, max_tokens=300)
            
            if not response.strip():
                print(f"Attempt {attempt + 1}: LLM returned empty response, retrying...")
                continue
                
            context, client_utterance = self._parse_dialogue_response(response)
            
            if context and client_utterance:
                # Generate MI therapist response
                therapist_response = self.generate_therapist_response(context, client_utterance)
                if therapist_response:
                    return context, client_utterance, therapist_response
                else:
                    print(f"Attempt {attempt + 1}: Failed to generate therapist response, retrying...")
            else:
                print(f"Attempt {attempt + 1}: Failed to parse dialogue, retrying...")
                
        # If all retries failed, raise an exception
        raise Exception("Failed to generate dialogue after maximum retries")
    
    def _parse_dialogue_response(self, response: str) -> Tuple[str, str]:
        """Parse the LLM response to extract context and client utterance."""
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            context = ""
            client_utterance = ""
            
            # Strategy 1: Look for exact "Context:" and "Client:" labels (with markdown formatting)
            for line in lines:
                line_clean = line.lower().strip()
                if line_clean.startswith('context:') or line_clean.startswith('**context:**'):
                    # Remove markdown and extract content
                    context = line.split(':', 1)[1].strip().replace('**', '').strip()
                elif line_clean.startswith('client:') or line_clean.startswith('**client:**'):
                    # Remove markdown, quotes, and extract content
                    client_utterance = line.split(':', 1)[1].strip().replace('**', '').strip().strip('"')
            
            # Strategy 2: If that failed, look for other patterns
            if not context or not client_utterance:
                for i, line in enumerate(lines):
                    line_clean = line.lower().replace('**', '')
                    # Look for context indicators
                    if any(word in line_clean for word in ['session', 'year-old', 'counseling', 'therapy', 'working', 'mother', 'father']) and not context:
                        context = line.replace('**', '').strip()
                    # Look for client speech indicators (quoted speech)
                    elif ('"' in line and any(phrase in line_clean for phrase in ['i ', 'my ', 'i\'m ', 'i don\'t', 'i think', 'honestly', 'look'])) and not client_utterance:
                        client_utterance = line.replace('**', '').strip().strip('"')
            
            # Clean up any remaining markdown
            context = context.replace('**', '').strip()
            client_utterance = client_utterance.replace('**', '').strip()
            
            print(f"DEBUG - Raw LLM response:\n{response}")
            print(f"DEBUG - Parsed context: '{context}'")
            print(f"DEBUG - Parsed client: '{client_utterance}'")
            
            return context, client_utterance
            
        except Exception as e:
            print(f"Error parsing dialogue response: {e}")
            return "", ""
    
    def generate_therapist_response(self, context: str, client_utterance: str) -> str:
        """Generate MI-adherent therapist response."""
        
        system_prompt = DialoguePair("", "", "", "", 0, 0, 0, {}, "").get_system_prompt()
        
        max_retries = 3
        for attempt in range(max_retries):
            prompt = f"""{system_prompt}

CLINICAL CONTEXT: {context}

CLIENT STATEMENT: "{client_utterance}"

CRITICAL: This client is showing some ambivalence or sustain talk about marijuana use. Your response must follow core MI principles:

DO NOT ASK ABOUT:
- Benefits of marijuana use
- What helps them when using
- Problems when NOT using  
- Reasons to continue using
- How marijuana makes them feel good

INSTEAD, FOCUS ON:
- Reflecting their concerns or conflicts
- Asking about problems FROM using
- Asking about goals that conflict with use
- Exploring their values as a parent/professional/partner
- Asking about times they didn't use successfully

Provide a motivational interviewing response that:
- Shows empathy and understanding
- Uses reflective listening (simple or complex reflection preferred)
- Gently guides toward CHANGE TALK, never sustain talk
- Avoids eliciting reasons to continue using
- Maintains collaborative stance
- Sounds natural and conversational

IMPORTANT: Provide ONLY the therapist's spoken response. Do not include any analysis, explanation, or commentary about the response.

THERAPIST RESPONSE:"""

            response = self.llm_client.generate_response(prompt, model=self.generation_model, max_tokens=200)
            
            if not response.strip():
                print(f"Therapist response attempt {attempt + 1}: Empty response, retrying...")
                continue
                
            # Clean the response to remove any commentary
            cleaned_response = self._clean_therapist_response(response)
            
            if cleaned_response and len(cleaned_response.split()) > 3:  # Ensure meaningful response
                return cleaned_response
            else:
                print(f"Therapist response attempt {attempt + 1}: Response too short or empty after cleaning, retrying...")
                
        # If all retries failed, return None to signal failure
        return None
    
    def _clean_therapist_response(self, response: str) -> str:
        """Clean therapist response to remove commentary or analysis."""
        response = response.strip()
        
        # Remove common LLM commentary patterns
        cleanup_patterns = [
            r'\n\nThis response:.*',  # Remove analysis starting with "This response:"
            r'\n\n\*.*',  # Remove bullet points starting with *
            r'\n\nThe response.*',   # Remove "The response..." explanations
            r'\n\nI.*',              # Remove "I used..." explanations
            r'\n\nNote:.*',          # Remove notes
        ]
        
        import re
        for pattern in cleanup_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)
        
        # Remove quotes if the entire response is wrapped in quotes
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        return response.strip()
    


class MITIEvaluatorLLM:
    """LLM specialized for evaluating MI fidelity using MITI 4.2.1 criteria."""
    
    def __init__(self, llm_client: LLMClient, evaluation_model: str = "llama3.2:latest"):
        self.llm_client = llm_client
        self.evaluation_model = evaluation_model
        
    def evaluate_dialogue(self, context: str, client_utterance: str, therapist_response: str) -> Tuple[float, float, Dict]:
        """Evaluate both realism and MI fidelity of dialogue pair."""
        
        realism_score = self.evaluate_realism(context, client_utterance)
        mi_score, miti_breakdown = self.evaluate_mi_fidelity(context, client_utterance, therapist_response)
        
        return realism_score, mi_score, miti_breakdown
    
    def evaluate_realism(self, context: str, client_utterance: str) -> float:
        """Evaluate how realistic the client utterance sounds."""
        
        prompt = f"""You are an expert in substance use counseling with 20+ years of clinical experience. Evaluate how realistic this client statement sounds.

CONTEXT: {context}
CLIENT STATEMENT: "{client_utterance}"

Rate the REALISM of this client statement on a continuous scale from 1.00 to 5.00 (use decimals for precision):

5.00 = Perfectly realistic - sounds exactly like something a real client would say
4.50 = Extremely realistic - very authentic with minor imperfections
4.00 = Very realistic - authentic but some artificial elements
3.50 = Moderately realistic - somewhat authentic but feels scripted
3.00 = Average realism - mix of authentic and artificial elements
2.50 = Below average - more artificial than authentic
2.00 = Unrealistic - sounds forced or generated
1.50 = Very unrealistic - clearly artificial
1.00 = Completely unrealistic - obviously generated

Consider:
- Natural speech patterns and word choice
- Emotional authenticity and ambivalence
- Appropriate resistance/change talk for the context
- Cultural and demographic authenticity
- Avoids overly clinical or textbook language

Provide your rating as a decimal number (e.g., 3.75, 4.25, 4.67):"""

        response = self.llm_client.generate_response(prompt, model=self.evaluation_model, max_tokens=100)
        
        try:
            # Extract decimal score from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response.strip())
            if numbers:
                score = float(numbers[0])
                return max(1.0, min(5.0, score))  # Clamp to 1-5 range
            else:
                return 3.0  # Default if no number found
        except:
            return 3.0  # Default to middle score
    
    def evaluate_mi_fidelity(self, context: str, client_utterance: str, therapist_response: str) -> Tuple[float, Dict]:
        """Evaluate MI fidelity using MITI 4.2.1 criteria."""
        
        prompt = f"""You are a certified MITI 4.2.1 coder with extensive training in Motivational Interviewing evaluation. Evaluate this therapist response for MI fidelity.

CONTEXT: {context}
CLIENT: "{client_utterance}"
THERAPIST: "{therapist_response}"

CRITICAL EVALUATION CRITERIA - CHANGE TALK vs SUSTAIN TALK:

The therapist response should be evaluated HARSHLY if it:
❌ Asks about benefits of marijuana use ("What helps you relax about pot?")
❌ Asks about problems when NOT using ("What happens when you don't use?") 
❌ Elicits sustain talk or reasons to continue using
❌ Explores how marijuana "helps" the client
❌ Asks about barriers to quitting

The therapist response should be rated HIGHLY if it:
✅ Reflects client concerns or conflicts about use
✅ Asks about problems FROM marijuana use
✅ Asks about goals that conflict with use
✅ Explores discrepancies between values and behavior
✅ Uses complex reflections that guide toward change

Using MITI 4.2.1 criteria, evaluate the therapist response on these Global Ratings using a continuous scale from 1.00 to 5.00 (use decimals for precision):

1. CULTIVATING CHANGE TALK: How well does the therapist encourage client language in favor of change? (Rate LOW if eliciting sustain talk)
2. SOFTENING SUSTAIN TALK: How well does the therapist avoid focus on reasons against change? (Rate LOW if asking about marijuana benefits)
3. PARTNERSHIP: How well does the therapist convey collaboration vs. expert role?
4. EMPATHY: How well does the therapist understand/grasp client perspective?

Use decimal precision (e.g., 3.25, 4.75, 4.15) to capture nuanced differences in quality.

REQUIRED FORMAT (provide exact decimal scores):
Cultivating_Change_Talk: [score 1.00-5.00]
Softening_Sustain_Talk: [score 1.00-5.00]
Partnership: [score 1.00-5.00]
Empathy: [score 1.00-5.00]
Primary_Behavior: [Simple_Reflection/Complex_Reflection/Question/Affirmation/etc]
MI_Adherent: [Yes/No]

Evaluation:"""

        response = self.llm_client.generate_response(prompt, model=self.evaluation_model, max_tokens=300)
        
        try:
            miti_breakdown = self._parse_miti_evaluation(response)
            overall_score = miti_breakdown.get('overall_score', 3.0)
            return overall_score, miti_breakdown
        except Exception as e:
            print(f"Error parsing MITI evaluation: {e}")
            print(f"Raw response: {response}")
            return 3.0, {'overall_score': 3.0, 'error': str(e), 'raw_response': response}
    
    def _parse_miti_evaluation(self, response: str) -> Dict:
        """Parse MITI evaluation response into structured data."""
        breakdown = {}
        
        try:
            lines = response.strip().split('\n')
            
            # Look for the specific format we requested
            score_mapping = {
                'cultivating_change_talk': 'cultivating_change_talk',
                'softening_sustain_talk': 'softening_sustain_talk', 
                'partnership': 'partnership',
                'empathy': 'empathy',
                'primary_behavior': 'primary_behavior',
                'mi_adherent': 'mi_adherent'
            }
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key_clean = key.strip().lower().replace(' ', '_')
                    value_clean = value.strip()
                    
                    # Map to our expected keys
                    for search_key, mapped_key in score_mapping.items():
                        if search_key in key_clean:
                            if mapped_key in ['cultivating_change_talk', 'softening_sustain_talk', 'partnership', 'empathy']:
                                # Extract decimal score with more precision
                                import re
                                numbers = re.findall(r'\d+\.?\d*', value_clean)
                                if numbers:
                                    score = float(numbers[0])
                                    breakdown[mapped_key] = max(1.0, min(5.0, score))  # Clamp to 1-5
                            else:
                                breakdown[mapped_key] = value_clean
                            break
            
            # Calculate derived scores with decimal precision
            cultivating = breakdown.get('cultivating_change_talk', 3.0)
            softening = breakdown.get('softening_sustain_talk', 3.0)
            partnership = breakdown.get('partnership', 3.0)
            empathy = breakdown.get('empathy', 3.0)
            
            breakdown['technical_score'] = round((cultivating + softening) / 2, 2)
            breakdown['relational_score'] = round((partnership + empathy) / 2, 2)
            breakdown['overall_score'] = round((breakdown['technical_score'] + breakdown['relational_score']) / 2, 2)
            
            # Ensure we have all required scores
            if not all(key in breakdown for key in ['cultivating_change_talk', 'softening_sustain_talk', 'partnership', 'empathy']):
                print(f"Warning: Missing some MITI scores. Found keys: {list(breakdown.keys())}")
                # Fill in missing scores with defaults
                for key in ['cultivating_change_talk', 'softening_sustain_talk', 'partnership', 'empathy']:
                    if key not in breakdown:
                        breakdown[key] = 3.0
                        
        except Exception as e:
            print(f"Error parsing MITI breakdown: {e}")
            breakdown = {
                'cultivating_change_talk': 3.0,
                'softening_sustain_talk': 3.0,
                'partnership': 3.0,
                'empathy': 3.0,
                'technical_score': 3.0,
                'relational_score': 3.0,
                'overall_score': 3.0,
                'parse_error': str(e)
            }
            
        return breakdown

class DialogueArchive:
    """Manages the archive of dialogue pairs with quality-based selection."""
    
    def __init__(self, archive_file: str = "mi_dialogue_archive.json", 
                 training_file: str = "mi_training_data.json",
                 top_k: int = 1000):
        self.archive_file = Path(archive_file)
        self.training_file = Path(training_file)
        self.top_k = top_k
        self.archive = []
        self.load_archive()
        
    def load_archive(self):
        """Load existing archive if it exists."""
        if self.archive_file.exists():
            try:
                with open(self.archive_file, 'r') as f:
                    data = json.load(f)
                    self.archive = [
                        DialoguePair(**item) for item in data
                    ]
                print(f"Loaded {len(self.archive)} dialogue pairs from archive")
            except Exception as e:
                print(f"Error loading archive: {e}")
                self.archive = []
    
    def add_dialogue_pair(self, pair: DialoguePair):
        """Add new dialogue pair to archive."""
        # Check for duplicates
        pair_hash = self._hash_dialogue(pair.client_utterance, pair.therapist_response)
        existing_hashes = {self._hash_dialogue(p.client_utterance, p.therapist_response) for p in self.archive}
        
        if pair_hash in existing_hashes:
            print("Duplicate dialogue detected, skipping...")
            return
            
        self.archive.append(pair)
        
        # Keep archive sorted by total score (descending)
        self.archive.sort(key=lambda x: x.total_score, reverse=True)
        
        # Trim archive if too large (keep top 2*top_k for diversity)
        if len(self.archive) > 2 * self.top_k:
            self.archive = self.archive[:2 * self.top_k]
            
        print(f"Added dialogue pair (Score: {pair.total_score:.2f}). Archive size: {len(self.archive)}")
        
    def get_top_pairs_for_training(self) -> List[Dict]:
        """Get top K pairs formatted for fine-tuning."""
        top_pairs = self.archive[:self.top_k]
        return [pair.to_json() for pair in top_pairs]
    
    def save_archive(self):
        """Save archive to file."""
        try:
            # Ensure directory exists
            self.archive_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.archive_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(pair) for pair in self.archive], f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(self.archive)} pairs to {self.archive_file}")
        except Exception as e:
            print(f"✗ Error saving archive: {e}")
    
    def save_training_data(self):
        """Save top pairs as training data."""
        try:
            # Ensure directory exists
            self.training_file.parent.mkdir(parents=True, exist_ok=True)
            
            training_data = self.get_top_pairs_for_training()
            with open(self.training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved top {len(training_data)} pairs as training data to {self.training_file}")
        except Exception as e:
            print(f"✗ Error saving training data: {e}")
    
    def get_stats(self) -> Dict:
        """Get archive statistics."""
        if not self.archive:
            return {"total": 0}
            
        scores = [pair.total_score for pair in self.archive]
        mi_scores = [pair.mi_fidelity_score for pair in self.archive]
        realism_scores = [pair.realism_score for pair in self.archive]
        
        return {
            "total": len(self.archive),
            "avg_total_score": sum(scores) / len(scores),
            "avg_mi_score": sum(mi_scores) / len(mi_scores),
            "avg_realism_score": sum(realism_scores) / len(realism_scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "top_k_size": min(self.top_k, len(self.archive))
        }
    
    def _hash_dialogue(self, client: str, therapist: str) -> str:
        """Create hash for dialogue pair to detect duplicates."""
        content = f"{client.lower().strip()}{therapist.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()

class EvolutionaryMIGenerator:
    """Main class orchestrating the evolutionary dialogue generation process."""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", 
                 generation_model: str = "wizardlm2:7b",
                 evaluation_model: str = "llama3.2:latest"):
        self.llm_client = LLMClient(llm_base_url)
        self.generator = DialogueGeneratorLLM(self.llm_client, generation_model)
        self.evaluator = MITIEvaluatorLLM(self.llm_client, evaluation_model)
        self.archive = DialogueArchive()
        self.generation_count = 0
        
        print(f"🤖 Using {generation_model} for generation, {evaluation_model} for evaluation")
        
    def generate_single_pair(self) -> Optional[DialoguePair]:
        """Generate and evaluate a single dialogue pair."""
        try:
            print(f"\n--- Generating Dialogue Pair #{self.generation_count + 1} ---")
            
            # Generate dialogue with retries
            context, client_utterance, therapist_response = self.generator.generate_dialogue_pair(self.archive.archive)
            
            print(f"Context: {context}")
            print(f"Client: {client_utterance}")
            print(f"Therapist: {therapist_response}")
            
            # Evaluate dialogue
            print("Evaluating dialogue...")
            realism_score, mi_score, miti_breakdown = self.evaluator.evaluate_dialogue(
                context, client_utterance, therapist_response
            )
            
            # Calculate total score (weighted combination)
            total_score = (realism_score * 0.3) + (mi_score * 0.7)  # Weight MI fidelity higher
            
            # Create dialogue pair
            pair_id = f"mi_pair_{int(time.time())}_{random.randint(1000, 9999)}"
            pair = DialoguePair(
                id=pair_id,
                client_utterance=client_utterance,
                therapist_response=therapist_response,
                context=context,
                mi_fidelity_score=mi_score,
                realism_score=realism_score,
                total_score=total_score,
                miti_breakdown=miti_breakdown,
                timestamp=datetime.now().isoformat()
            )
            
            print(f"Scores - Realism: {realism_score:.2f}, MI Fidelity: {mi_score:.2f}, Total: {total_score:.2f}")
            
            self.generation_count += 1
            return pair
            
        except Exception as e:
            print(f"Error generating dialogue pair: {e}")
            print("Skipping this generation and continuing...")
            return None
    
    def run_continuous_generation(self, target_generations: int = float('inf'), save_interval: int = 10):
        """Run continuous generation process."""
        print(f"Starting evolutionary MI dialogue generation for cannabis counseling...")
        if target_generations == float('inf'):
            print(f"Running continuously until interrupted (Ctrl+C to stop)")
        else:
            print(f"Target generations: {target_generations}")
        print(f"Current archive size: {len(self.archive.archive)}")
        
        generated_count = 0
        
        try:
            while generated_count < target_generations:
                pair = self.generate_single_pair()
                
                if pair:
                    self.archive.add_dialogue_pair(pair)
                    generated_count += 1
                    
                    # Save both archive AND training data immediately after each generation
                    self.archive.save_archive()
                    self.archive.save_training_data()
                    
                    # Periodic stats reporting
                    if generated_count % save_interval == 0:
                        self.print_stats()
                    elif generated_count % 3 == 0:  # Show progress more frequently
                        print(f"Generated {generated_count} pairs so far...")
                        
                # Small delay to prevent overwhelming the LLM
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
        finally:
            # Final save
            print("\nSaving final results...")
            self.archive.save_archive()
            self.archive.save_training_data()
            self.print_stats()
    
    def print_stats(self):
        """Print current archive statistics."""
        stats = self.archive.get_stats()
        print(f"\n=== ARCHIVE STATISTICS ===")
        print(f"Total pairs: {stats['total']}")
        if stats['total'] > 0:
            print(f"Average total score: {stats['avg_total_score']:.2f}")
            print(f"Average MI fidelity: {stats['avg_mi_score']:.2f}")
            print(f"Average realism: {stats['avg_realism_score']:.2f}")
            print(f"Best score: {stats['best_score']:.2f}")
            print(f"Worst score: {stats['worst_score']:.2f}")
            print(f"Training data size: {stats['top_k_size']}")
        print("========================\n")
    
    def run_single_generation(self):
        """Generate and evaluate a single dialogue pair for testing."""
        pair = self.generate_single_pair()
        if pair:
            self.archive.add_dialogue_pair(pair)
            self.archive.save_archive()
            self.archive.save_training_data()
            self.print_stats()
            return pair
        return None

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evolutionary MI Dialogue Generator")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="Generation mode: single pair or continuous")
    parser.add_argument("--target", type=int, default=0,
                       help="Target number of generations (0 = infinite for continuous mode)")
    parser.add_argument("--llm-url", default="http://localhost:11434",
                       help="Local LLM API URL")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save interval for continuous mode")
    # Add the missing arguments:
    parser.add_argument("--generation-model", default="wizardlm2:7b",
                       help="Model to use for dialogue generation")
    parser.add_argument("--evaluation-model", default="llama3.2:latest", 
                       help="Model to use for dialogue evaluation")
    
    args = parser.parse_args()
    
    # Initialize generator - fix the attribute names to match the argument names
    print("Initializing Evolutionary MI Dialogue Generator...")
    generator = EvolutionaryMIGenerator(
        args.llm_url, 
        args.generation_model,  # This will now work
        args.evaluation_model   # This will now work
    )
    
    if args.mode == "single":
        print("Running single generation...")
        pair = generator.run_single_generation()
        if pair:
            print("\n✓ Successfully generated and archived dialogue pair")
        else:
            print("\n✗ Failed to generate dialogue pair")
    
    elif args.mode == "continuous":
        target = float('inf') if args.target == 0 else args.target
        print(f"Running continuous generation (target: {'infinite' if target == float('inf') else target})...")
        generator.run_continuous_generation(target, args.save_interval)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()