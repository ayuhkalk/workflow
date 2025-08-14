# agents/agent_coordinator.py
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .document_agent import DocumentAgent
from .fmea_agent import FMEAAgent
from .chat_agent import ChatAgent
from .learning_agent import LearningAgent
from .report_generator_agent import ReportGeneratorAgent

class AgentCoordinator:
    """
    Central coordinator that orchestrates communication between all agents
    Implements the AgentCoordinator from the class diagram
    """
    
    def __init__(self):
        self.document_agent = DocumentAgent()
        self.fmea_agent = FMEAAgent()
        self.chat_agent = ChatAgent()
        self.learning_agent = LearningAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def run_pipeline(self, uploaded_files: List, user_data: Dict = None) -> Dict[str, Any]:
        """
        Main pipeline execution as per use case diagram
        """
        try:
            # Step 1: Process documents
            self.logger.info("Starting document processing...")
            processed_data = self.process_documents(uploaded_files)
            
            # Step 2: Generate initial FMEA
            self.logger.info("Generating FMEA...")
            fmea_data = self.generate_fmea(processed_data, user_data)
            
            # Step 3: Apply any learning improvements
            improved_fmea = self.apply_learning_improvements(fmea_data)
            
            return {
                'success': True,
                'processed_data': processed_data,
                'fmea_data': improved_fmea,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def process_documents(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Coordinate document processing through DocumentAgent
        """
        try:
            all_extracted_data = {
                'extracted_text': '',
                'tables': [],
                'failure_modes': [],
                'technical_specs': {},
                'historical_fmea': None
            }
            
            for file in uploaded_files:
                self.logger.info(f"Processing file: {file.name}")
                
                # Extract data based on file type
                if file.name.endswith('.pdf'):
                    extracted = self.document_agent.extract_text(file)
                    all_extracted_data['extracted_text'] += extracted + "\n"
                    
                    # Try to extract tables from PDF
                    tables = self.document_agent.extract_tables_from_pdf(file)
                    all_extracted_data['tables'].extend(tables)
                    
                elif file.name.endswith(('.xlsx', '.csv')):
                    df = self.document_agent.read_excel(file)
                    all_extracted_data['tables'].append(df)
                    
                    # Check if this looks like historical FMEA data
                    if self._is_fmea_format(df):
                        all_extracted_data['historical_fmea'] = df
                        
                elif file.name.endswith(('.png', '.jpg', '.jpeg')):
                    # Extract text from images using OCR
                    text = self.document_agent.extract_text_from_image(file)
                    all_extracted_data['extracted_text'] += text + "\n"
                
                elif file.name.endswith('.txt'):
                    text = str(file.read(), "utf-8")
                    all_extracted_data['extracted_text'] += text + "\n"
            
            # Extract potential failure modes from all text
            if all_extracted_data['extracted_text']:
                failure_modes = self.document_agent.extract_failure_modes(
                    all_extracted_data['extracted_text']
                )
                all_extracted_data['failure_modes'] = failure_modes
                
                # Extract technical specifications
                tech_specs = self.document_agent.extract_technical_specs(
                    all_extracted_data['extracted_text']
                )
                all_extracted_data['technical_specs'] = tech_specs
            
            self.logger.info("Document processing completed successfully")
            return all_extracted_data
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise
    
    def generate_fmea(self, processed_data: Dict, user_data: Dict = None) -> pd.DataFrame:
        """
        Generate FMEA using FMEAAgent
        """
        try:
            self.logger.info("Generating FMEA table...")
            
            # Prepare context for FMEA generation
            context = {
                'extracted_text': processed_data.get('extracted_text', ''),
                'failure_modes': processed_data.get('failure_modes', []),
                'technical_specs': processed_data.get('technical_specs', {}),
                'historical_fmea': processed_data.get('historical_fmea'),
                'user_preferences': user_data or {}
            }
            
            # Generate initial FMEA table
            fmea_df = self.fmea_agent.generate_initial_fmea(context)
            
            # If we have historical FMEA data, use it to improve suggestions
            if processed_data.get('historical_fmea') is not None:
                fmea_df = self.fmea_agent.improve_with_historical_data(
                    fmea_df, processed_data['historical_fmea']
                )
            
            self.logger.info(f"Generated FMEA with {len(fmea_df)} entries")
            return fmea_df
            
        except Exception as e:
            self.logger.error(f"FMEA generation failed: {str(e)}")
            raise
    
    def chat_with_fmea(self, user_message: str, current_fmea: pd.DataFrame) -> Dict[str, Any]:
        """
        Handle chat interactions about FMEA
        """
        try:
            # Pass to ChatAgent for processing
            response = self.chat_agent.guide_user(user_message, current_fmea)
            
            # If the chat resulted in FMEA modifications, record for learning
            if 'updated_fmea' in response:
                self.learning_agent.store_feedback(
                    user_message, 
                    current_fmea, 
                    response['updated_fmea']
                )
                
                # Train reinforcement learner
                self.reinforcement_learner.record_changes(
                    current_fmea, 
                    response['updated_fmea']
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Chat interaction failed: {str(e)}")
            return {
                'message': f"Sorry, I encountered an error: {str(e)}",
                'error': True
            }
    
    def apply_learning_improvements(self, fmea_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply improvements based on learning agent data
        """
        try:
            # Get learning-based improvements
            improvements = self.learning_agent.get_improved_suggestions(fmea_data)
            
            if improvements:
                self.logger.info("Applying learning-based improvements")
                return self.fmea_agent.apply_improvements(fmea_data, improvements)
            
            return fmea_data
            
        except Exception as e:
            self.logger.error(f"Learning improvements failed: {str(e)}")
            return fmea_data
    
    def analyze_high_risk_items(self, fmea_data: pd.DataFrame) -> str:
        """
        Analyze high-risk items in FMEA
        """
        try:
            return self.fmea_agent.analyze_high_risk_items(fmea_data)
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def suggest_improvements(self, fmea_data: pd.DataFrame) -> str:
        """
        Suggest improvements for FMEA
        """
        try:
            return self.fmea_agent.suggest_improvements(fmea_data)
        except Exception as e:
            return f"Suggestion generation failed: {str(e)}"
    
    def validate_fmea(self, fmea_data: pd.DataFrame) -> str:
        """
        Validate FMEA completeness and consistency
        """
        try:
            return self.fmea_agent.validate_fmea(fmea_data)
        except Exception as e:
            return f"Validation failed: {str(e)}"
    
    def pass_to_agent(self, agent: str, data: Dict) -> Dict:
        """
        Route data to specific agent
        """
        try:
            if agent == "document":
                return self.document_agent.extract_data(data)
            elif agent == "fmea":
                return self.fmea_agent.create_fmea_table(data)
            elif agent == "chat":
                return self.chat_agent.guide_user(data.get('message', ''), data.get('fmea'))
            elif agent == "learning":
                return self.learning_agent.store_feedback(
                    data.get('feedback'), 
                    data.get('original'), 
                    data.get('modified')
                )
            else:
                raise ValueError(f"Unknown agent: {agent}")
                
        except Exception as e:
            self.logger.error(f"Agent routing failed: {str(e)}")
            return {'error': str(e)}
    
    def _is_fmea_format(self, df: pd.DataFrame) -> bool:
        """
        Check if dataframe looks like FMEA format
        """
        fmea_columns = ['failure mode', 'effect', 'cause', 'severity', 'occurrence', 'detection', 'rpn']
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Check if at least 4 of the typical FMEA columns are present
        matches = sum(1 for col in fmea_columns if any(col in df_col for df_col in df_columns_lower))
        return matches >= 4
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        """
        return {
            'document_agent': self.document_agent.get_status(),
            'fmea_agent': self.fmea_agent.get_status(),
            'chat_agent': self.chat_agent.get_status(),
            'learning_agent': self.learning_agent.get_status(),
            'timestamp': datetime.now()
        }