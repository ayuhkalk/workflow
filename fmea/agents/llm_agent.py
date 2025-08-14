# agents/llm_agent.py
import requests
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class OllamaLLMAgent:
    """
    Integration with Ollama local LLM for enhanced FMEA generation and chat
    Supports Ollama 3.2 1B and other Ollama models
    """
    
    def __init__(self, model_name: str = "llama3.2:1b", host: str = "localhost", port: int = 8501):
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
        
        # FMEA-specific system prompts
        self.system_prompts = {
            'fmea_generation': """You are an expert FMEA (Failure Mode and Effects Analysis) analyst specializing in mechanical and electrical systems, particularly CNC machines and manufacturing equipment. 

Your role is to:
1. Analyze technical documents and identify potential failure modes
2. Suggest appropriate Severity, Occurrence, and Detection ratings (1-10 scale)
3. Provide clear, concise failure descriptions
4. Recommend detection methods and preventive actions

Guidelines:
- Severity: 1=no impact, 10=catastrophic/safety issue
- Occurrence: 1=remote probability, 10=almost certain
- Detection: 1=almost certain to detect, 10=cannot detect
- Be specific and technical in your responses
- Focus on actionable insights""",

            'chat_assistant': """You are a helpful FMEA chat assistant. Help users understand and improve their Failure Mode and Effects Analysis.

You can:
- Explain FMEA concepts (RPN, Severity, Occurrence, Detection)
- Analyze risk data and provide insights
- Suggest improvements for high-risk items
- Help modify FMEA entries
- Answer questions about failure modes and risk assessment

Be conversational but professional. Provide clear, actionable advice.""",

            'document_analysis': """You are a technical document analyzer specializing in extracting failure-related information from manuals, service bulletins, and technical documentation.

Extract and identify:
- Component names and systems
- Potential failure modes
- Failure causes and effects
- Maintenance procedures
- Safety warnings and precautions
- Technical specifications

Present information in a structured, clear format suitable for FMEA analysis."""
        }
        
        # Check if Ollama is available
        self.is_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.warning("Ollama service not available")
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if not any(self.model_name in name for name in model_names):
                self.logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                # Try to pull the model
                self._pull_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ollama availability check failed: {str(e)}")
            return False
    
    def _pull_model(self) -> bool:
        """Pull the specified model if not available"""
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                self.logger.info(f"Model {self.model_name} pulled successfully")
                return True
            else:
                self.logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model pull failed: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, context: str = "chat_assistant", 
                         max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using Ollama LLM"""
        try:
            if not self.is_available:
                return {
                    'success': False,
                    'message': 'LLM service not available. Falling back to rule-based response.',
                    'fallback': True
                }
            
            # Prepare the full prompt with system context
            system_prompt = self.system_prompts.get(context, self.system_prompts['chat_assistant'])
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'message': result.get('response', '').strip(),
                    'model': self.model_name,
                    'tokens_used': result.get('eval_count', 0),
                    'generation_time': result.get('eval_duration', 0) / 1e9  # Convert to seconds
                }
            else:
                self.logger.error(f"LLM request failed: {response.text}")
                return {
                    'success': False,
                    'message': 'LLM request failed. Using fallback response.',
                    'fallback': True
                }
                
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return {
                'success': False,
                'message': f'LLM error: {str(e)}',
                'fallback': True
            }
    
    def analyze_document_for_fmea(self, document_text: str) -> Dict[str, Any]:
        """Analyze document text to extract FMEA-relevant information"""
        try:
            prompt = f"""Analyze the following technical document and extract information relevant for FMEA (Failure Mode and Effects Analysis):

Document Text:
{document_text[:3000]}  # Limit text to avoid token limits

Please identify and extract:
1. Components and systems mentioned
2. Potential failure modes
3. Failure causes and effects
4. Maintenance procedures
5. Safety warnings

Format your response as a structured analysis suitable for creating FMEA entries."""

            response = self.generate_response(prompt, context="document_analysis", max_tokens=1500)
            
            if response['success']:
                # Parse the response to extract structured data
                parsed_data = self._parse_document_analysis(response['message'])
                return {
                    'success': True,
                    'analysis': response['message'],
                    'structured_data': parsed_data,
                    'tokens_used': response.get('tokens_used', 0)
                }
            else:
                return response
                
        except Exception as e:
            self.logger.error(f"Document analysis failed: {str(e)}")
            return {
                'success': False,
                'message': f'Document analysis error: {str(e)}'
            }
    
    def generate_fmea_entries(self, components: List[str], context: str = "") -> List[Dict[str, Any]]:
        """Generate FMEA entries for given components"""
        try:
            entries = []
            
            for component in components:
                prompt = f"""Generate FMEA entries for the component: {component}

Context: {context}

For this component, provide 2-3 potential failure modes with:
1. Failure Mode description
2. Effect of failure
3. Potential cause
4. Suggested detection method
5. Recommended Severity rating (1-10)
6. Recommended Occurrence rating (1-10)  
7. Recommended Detection rating (1-10)

Format each entry clearly with the above fields."""

                response = self.generate_response(prompt, context="fmea_generation", max_tokens=800)
                
                if response['success']:
                    component_entries = self._parse_fmea_entries(response['message'], component)
                    entries.extend(component_entries)
            
            return entries
            
        except Exception as e:
            self.logger.error(f"FMEA entry generation failed: {str(e)}")
            return []
    
    def chat_about_fmea(self, user_message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle chat interactions about FMEA data"""
        try:
            # Prepare context about current FMEA
            fmea_summary = self._create_fmea_summary(fmea_data)
            
            prompt = f"""Current FMEA Summary:
{fmea_summary}

User Question: {user_message}

Please provide a helpful response about the FMEA. If the user is asking to modify data, provide specific guidance on what changes to make."""

            response = self.generate_response(prompt, context="chat_assistant", max_tokens=800)
            
            # Check if response contains modification instructions
            modifications = self._extract_modifications(response.get('message', ''), fmea_data)
            
            result = {
                'success': response['success'],
                'message': response.get('message', ''),
                'model': self.model_name
            }
            
            if modifications:
                result['modifications'] = modifications
                result['updated_fmea'] = self._apply_modifications(fmea_data, modifications)
            
            return result
            
        except Exception as e:
            self.logger.error(f"FMEA chat failed: {str(e)}")
            return {
                'success': False,
                'message': f'Chat error: {str(e)}'
            }
    
    def suggest_improvements(self, fmea_data: pd.DataFrame) -> str:
        """Generate improvement suggestions for FMEA"""
        try:
            high_risk_items = fmea_data[fmea_data['RPN'] > 100] if 'RPN' in fmea_data.columns else pd.DataFrame()
            
            if high_risk_items.empty:
                return "No high-risk items identified. FMEA appears to be well-managed."
            
            # Create summary of high-risk items
            risk_summary = []
            for _, row in high_risk_items.head(5).iterrows():
                risk_summary.append(f"- {row.get('Component', 'Unknown')}: {row.get('Failure Mode', 'Unknown')} (RPN: {row.get('RPN', 'Unknown')})")
            
            prompt = f"""Analyze these high-risk FMEA items and provide improvement suggestions:

High-Risk Items:
{chr(10).join(risk_summary)}

Please provide:
1. Specific actions to reduce Severity, Occurrence, or Detection ratings
2. Preventive measures that could be implemented
3. Monitoring and detection improvements
4. Prioritization recommendations

Focus on practical, actionable suggestions."""

            response = self.generate_response(prompt, context="fmea_generation", max_tokens=1000)
            return response.get('message', 'Unable to generate suggestions at this time.')
            
        except Exception as e:
            self.logger.error(f"Improvement suggestions failed: {str(e)}")
            return f'Error generating suggestions: {str(e)}'
    
    def _parse_document_analysis(self, analysis_text: str) -> Dict[str, List[str]]:
        """Parse LLM document analysis into structured data"""
        try:
            # Simple parsing - in production, you might want more sophisticated NLP
            structured_data = {
                'components': [],
                'failure_modes': [],
                'causes': [],
                'effects': [],
                'detection_methods': []
            }
            
            lines = analysis_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if 'component' in line.lower() and ':' in line:
                    current_section = 'components'
                elif 'failure' in line.lower() and 'mode' in line.lower():
                    current_section = 'failure_modes'
                elif 'cause' in line.lower():
                    current_section = 'causes'
                elif 'effect' in line.lower():
                    current_section = 'effects'
                elif 'detection' in line.lower():
                    current_section = 'detection_methods'
                elif line.startswith('-') or line.startswith('*'):
                    if current_section:
                        item = line.lstrip('-* ').strip()
                        if item and len(item) > 3:
                            structured_data[current_section].append(item)
            
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Document analysis parsing failed: {str(e)}")
            return {}
    
    def _parse_fmea_entries(self, llm_response: str, component: str) -> List[Dict[str, Any]]:
        """Parse LLM response into FMEA entry format"""
        try:
            entries = []
            # Simple parsing logic - can be enhanced with more sophisticated NLP
            
            # Split response into potential entries
            sections = llm_response.split('\n\n')
            
            for section in sections:
                if 'failure mode' in section.lower():
                    entry = {
                        'Component': component,
                        'Failure Mode': self._extract_field(section, 'failure mode'),
                        'Effect': self._extract_field(section, 'effect'),
                        'Cause': self._extract_field(section, 'cause'),
                        'Detection Method': self._extract_field(section, 'detection'),
                        'Severity': self._extract_rating(section, 'severity'),
                        'Occurrence': self._extract_rating(section, 'occurrence'),
                        'Detection': self._extract_rating(section, 'detection rating'),
                        'Source': f'LLM Generated ({self.model_name})'
                    }
                    
                    # Calculate RPN
                    entry['RPN'] = entry['Severity'] * entry['Occurrence'] * entry['Detection']
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            self.logger.error(f"FMEA entry parsing failed: {str(e)}")
            return []
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract specific field from LLM response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if field_name.lower() in line.lower() and ':' in line:
                    return line.split(':', 1)[1].strip()
            return 'To be determined'
        except:
            return 'To be determined'
    
    def _extract_rating(self, text: str, rating_name: str) -> int:
        """Extract numeric rating from LLM response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if rating_name.lower() in line.lower() and ':' in line:
                    rating_text = line.split(':', 1)[1].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d+', rating_text)
                    if numbers:
                        rating = int(numbers[0])
                        return max(1, min(10, rating))  # Ensure 1-10 range
            return 5  # Default rating
        except:
            return 5
    
    def _create_fmea_summary(self, fmea_data: pd.DataFrame) -> str:
        """Create a summary of FMEA data for LLM context"""
        try:
            if fmea_data.empty:
                return "No FMEA data available."
            
            summary = f"FMEA contains {len(fmea_data)} entries.\n"
            
            if 'RPN' in fmea_data.columns:
                avg_rpn = fmea_data['RPN'].mean()
                high_risk_count = len(fmea_data[fmea_data['RPN'] > 100])
                summary += f"Average RPN: {avg_rpn:.1f}, High-risk items: {high_risk_count}\n"
            
            if 'Component' in fmea_data.columns:
                components = fmea_data['Component'].value_counts().head(5)
                summary += f"Main components: {', '.join(components.index.tolist())}\n"
            
            return summary
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"
    
    def _extract_modifications(self, llm_response: str, fmea_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract modification instructions from LLM response"""
        # This would need more sophisticated NLP to properly extract modification instructions
        # For now, return empty list - can be enhanced based on specific LLM response patterns
        return []
    
    def _apply_modifications(self, fmea_data: pd.DataFrame, modifications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply modifications to FMEA data"""
        # Apply the modifications returned by _extract_modifications
        modified_data = fmea_data.copy()
        # Implementation depends on modification format
        return modified_data
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM agent status"""
        return {
            'name': 'OllamaLLMAgent',
            'model': self.model_name,
            'available': self.is_available,
            'base_url': self.base_url,
            'active': True
        }


# Enhanced integration with existing agents
class EnhancedFMEAAgent:
    """Enhanced FMEA Agent with LLM integration"""
    
    def __init__(self):
        from .fmea_agent import FMEAAgent  # Import original FMEA agent
        self.base_agent = FMEAAgent()
        self.llm_agent = OllamaLLMAgent()
        self.logger = logging.getLogger(__name__)
    
    def generate_initial_fmea(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate FMEA with LLM enhancement"""
        try:
            # First, use the base agent
            base_fmea = self.base_agent.generate_initial_fmea(context)
            
            # If LLM is available, enhance with LLM analysis
            if self.llm_agent.is_available and context.get('extracted_text'):
                llm_analysis = self.llm_agent.analyze_document_for_fmea(context['extracted_text'])
                
                if llm_analysis['success']:
                    # Generate additional entries based on LLM analysis
                    structured_data = llm_analysis.get('structured_data', {})
                    components = structured_data.get('components', [])
                    
                    if components:
                        llm_entries = self.llm_agent.generate_fmea_entries(
                            components[:5],  # Limit to 5 components to avoid too many entries
                            context.get('extracted_text', '')[:1000]
                        )
                        
                        if llm_entries:
                            llm_df = pd.DataFrame(llm_entries)
                            # Combine with base FMEA
                            base_fmea = pd.concat([base_fmea, llm_df], ignore_index=True)
            
            return base_fmea
            
        except Exception as e:
            self.logger.error(f"Enhanced FMEA generation failed: {str(e)}")
            # Fallback to base agent
            return self.base_agent.generate_initial_fmea(context)


class EnhancedChatAgent:
    """Enhanced Chat Agent with LLM integration"""
    
    def __init__(self):
        from .chat_agent import ChatAgent  # Import original chat agent
        self.base_agent = ChatAgent()
        self.llm_agent = OllamaLLMAgent()
        self.logger = logging.getLogger(__name__)
    
    def guide_user(self, user_message: str, current_fmea: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced user guidance with LLM"""
        try:
            # If LLM is available, use it for more natural responses
            if self.llm_agent.is_available:
                llm_response = self.llm_agent.chat_about_fmea(user_message, current_fmea)
                
                if llm_response['success']:
                    return llm_response
            
            # Fallback to base agent
            return self.base_agent.guide_user(user_message, current_fmea)
            
        except Exception as e:
            self.logger.error(f"Enhanced chat failed: {str(e)}")
            # Fallback to base agent
            return self.base_agent.guide_user(user_message, current_fmea)