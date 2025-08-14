# agents/chat_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import re
import json
from datetime import datetime

class ChatAgent:
    """
    Interactive chat agent that guides users through FMEA creation and refinement
    Implements the ChatAgent from the class diagram
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.context_memory = {}
        
        # Initialize chat capabilities
        self.setup_chat_handlers()
        self.setup_fmea_knowledge()
    
    def setup_chat_handlers(self):
        """Setup handlers for different types of chat interactions"""
        self.handlers = {
            'analyze': self.handle_analysis_request,
            'modify': self.handle_modification_request,
            'add': self.handle_add_request,
            'delete': self.handle_delete_request,
            'explain': self.handle_explanation_request,
            'suggest': self.handle_suggestion_request,
            'validate': self.handle_validation_request,
            'export': self.handle_export_request
        }
        
        # Command patterns for intent recognition
        self.intent_patterns = {
            'analyze': [
                r'analyze|analysis|examine|review|assess',
                r'what.*risk|high.*risk|critical',
                r'show.*statistics|stats|summary'
            ],
            'modify': [
                r'change|modify|update|edit|alter',
                r'set.*severity|set.*occurrence|set.*detection',
                r'rpn.*to|rating.*to'
            ],
            'add': [
                r'add|create|new|insert',
                r'another.*entry|more.*failure',
                r'include.*component'
            ],
            'delete': [
                r'delete|remove|eliminate',
                r'get.*rid|take.*out'
            ],
            'explain': [
                r'explain|what.*mean|help.*understand',
                r'why|how.*work|what.*is'
            ],
            'suggest': [
                r'suggest|recommend|advice|improve',
                r'what.*should|better.*way'
            ],
            'validate': [
                r'validate|check|verify|correct',
                r'is.*right|any.*error|problem'
            ],
            'export': [
                r'export|download|save|generate.*report',
                r'excel|csv|file'
            ]
        }
    
    def setup_fmea_knowledge(self):
        """Setup FMEA-specific knowledge for chat responses"""
        self.fmea_knowledge = {
            'severity_scale': {
                1: "No impact - failure has no noticeable effect",
                2: "Very minor - slight customer dissatisfaction", 
                3: "Minor - minor customer dissatisfaction",
                4: "Very low - low customer dissatisfaction",
                5: "Low - low customer dissatisfaction with warning",
                6: "Moderate - moderate customer dissatisfaction",
                7: "High - high customer dissatisfaction",
                8: "Very high - very high customer dissatisfaction", 
                9: "Hazardous - potential safety issue",
                10: "Catastrophic - safety issue, non-compliance"
            },
            'occurrence_scale': {
                1: "Remote - failure rate < 1 in 1,500,000",
                2: "Very low - failure rate 1 in 150,000",
                3: "Low - failure rate 1 in 15,000", 
                4: "Moderately low - failure rate 1 in 2,000",
                5: "Moderate - failure rate 1 in 400",
                6: "Moderately high - failure rate 1 in 80",
                7: "High - failure rate 1 in 20",
                8: "Very high - failure rate 1 in 8",
                9: "Extremely high - failure rate 1 in 3",
                10: "Almost certain - failure rate > 1 in 2"
            },
            'detection_scale': {
                1: "Almost certain - defect will almost certainly be detected",
                2: "Very high - very high probability of detection",
                3: "High - high probability of detection",
                4: "Moderately high - moderately high probability",
                5: "Moderate - moderate probability of detection", 
                6: "Low - low probability of detection",
                7: "Very low - very low probability of detection",
                8: "Remote - remote probability of detection",
                9: "Very remote - very remote probability",
                10: "Almost impossible - defect will not be detected"
            }
        }
    
    def guide_user(self, user_message: str, current_fmea: pd.DataFrame) -> Dict[str, Any]:
        """Main method to guide user through FMEA interaction"""
        try:
            # Store conversation context
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_message': user_message,
                'fmea_state': len(current_fmea) if not current_fmea.empty else 0
            })
            
            # Detect user intent
            intent = self.detect_intent(user_message)
            
            # Handle the request based on intent
            if intent in self.handlers:
                response = self.handlers[intent](user_message, current_fmea)
            else:
                response = self.handle_general_query(user_message, current_fmea)
            
            # Store response in conversation history
            self.conversation_history[-1]['assistant_response'] = response['message']
            
            return response
            
        except Exception as e:
            self.logger.error(f"Chat guidance failed: {str(e)}")
            return {
                'message': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your request.",
                'error': True
            }
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return 'general'
    
    def handle_analysis_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests for FMEA analysis"""
        if fmea_data.empty:
            return {
                'message': "I can't analyze the FMEA because no data is available. Please generate or upload FMEA data first.",
                'action_required': 'generate_fmea'
            }
        
        analysis = self.generate_fmea_analysis(fmea_data)
        
        return {
            'message': f"Here's the FMEA analysis:\n\n{analysis}",
            'analysis_data': analysis
        }
    
    def handle_modification_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests to modify FMEA entries"""
        if fmea_data.empty:
            return {
                'message': "There's no FMEA data to modify. Please generate FMEA data first.",
                'action_required': 'generate_fmea'
            }
        
        # Extract modification details from message
        modification = self.parse_modification_request(message, fmea_data)
        
        if modification:
            updated_fmea = self.apply_modification(fmea_data, modification)
            return {
                'message': f"I've updated the FMEA as requested: {modification['description']}",
                'updated_fmea': updated_fmea,
                'modification_applied': modification
            }
        else:
            return {
                'message': "I understand you want to modify the FMEA, but I need more specific details. Please specify:\n"
                          "• Which component or entry to modify\n"
                          "• What value to change (Severity, Occurrence, Detection, etc.)\n"
                          "• The new value\n\n"
                          "Example: 'Change the severity of Motor bearing failure to 8'",
                'action_required': 'clarification'
            }
    
    def handle_add_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests to add new FMEA entries"""
        # Check if user provided enough information
        entry_info = self.parse_add_request(message)
        
        if entry_info and all(key in entry_info for key in ['component', 'failure_mode']):
            new_entry = self.create_fmea_entry(entry_info)
            updated_fmea = pd.concat([fmea_data, pd.DataFrame([new_entry])], ignore_index=True)
            
            return {
                'message': f"I've added a new FMEA entry for {entry_info['component']} - {entry_info['failure_mode']}. "
                          f"Please review and adjust the ratings as needed.",
                'updated_fmea': updated_fmea,
                'new_entry': new_entry
            }
        else:
            return {
                'message': "I'd be happy to add a new FMEA entry! Please provide:\n"
                          "• Component name\n"
                          "• Failure mode\n"
                          "• Effect (optional)\n"
                          "• Cause (optional)\n\n"
                          "Example: 'Add motor overheating failure for spindle motor'",
                'action_required': 'entry_details'
            }
    
    def handle_delete_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests to delete FMEA entries"""
        if fmea_data.empty:
            return {
                'message': "There are no FMEA entries to delete.",
                'action_required': 'generate_fmea'
            }
        
        # Parse which entry to delete
        delete_info = self.parse_delete_request(message, fmea_data)
        
        if delete_info:
            updated_fmea = fmea_data.drop(delete_info['index']).reset_index(drop=True)
            return {
                'message': f"I've deleted the entry: {delete_info['description']}",
                'updated_fmea': updated_fmea,
                'deleted_entry': delete_info
            }
        else:
            # Show available entries for deletion
            entries_list = "\n".join([f"{i}: {row['Component']} - {row['Failure Mode']}" 
                                    for i, (_, row) in enumerate(fmea_data.iterrows())])
            return {
                'message': f"Please specify which entry to delete by number or description:\n\n{entries_list}",
                'action_required': 'specify_entry'
            }
    
    def handle_explanation_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests for explanations about FMEA concepts"""
        explanation_topic = self.identify_explanation_topic(message)
        
        explanations = {
            'severity': self.explain_severity(),
            'occurrence': self.explain_occurrence(), 
            'detection': self.explain_detection(),
            'rpn': self.explain_rpn(),
            'fmea': self.explain_fmea_general()
        }
        
        if explanation_topic in explanations:
            return {
                'message': explanations[explanation_topic],
                'explanation_topic': explanation_topic
            }
        else:
            return {
                'message': "I can explain various FMEA concepts:\n"
                          "• **Severity** - Impact of failure on customer\n"
                          "• **Occurrence** - Likelihood of failure happening\n"
                          "• **Detection** - Probability of detecting failure before it reaches customer\n"
                          "• **RPN** - Risk Priority Number calculation\n"
                          "• **FMEA** - General FMEA methodology\n\n"
                          "What would you like me to explain?",
                'available_topics': list(explanations.keys())
            }
    
    def handle_suggestion_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests for suggestions and improvements"""
        if fmea_data.empty:
            return {
                'message': "I need FMEA data to provide suggestions. Please generate FMEA data first.",
                'action_required': 'generate_fmea'
            }
        
        suggestions = self.generate_suggestions(fmea_data)
        
        return {
            'message': f"Here are my suggestions for improving your FMEA:\n\n{suggestions}",
            'suggestions': suggestions
        }
    
    def handle_validation_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests to validate FMEA"""
        if fmea_data.empty:
            return {
                'message': "There's no FMEA data to validate. Please generate FMEA data first.",
                'action_required': 'generate_fmea'
            }
        
        validation_results = self.validate_fmea_data(fmea_data)
        
        return {
            'message': f"FMEA Validation Results:\n\n{validation_results}",
            'validation_results': validation_results
        }
    
    def handle_export_request(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle requests to export FMEA data"""
        if fmea_data.empty:
            return {
                'message': "There's no FMEA data to export. Please generate FMEA data first.",
                'action_required': 'generate_fmea'
            }
        
        return {
            'message': "I can help you export the FMEA data. Please go to the 'Report Export' page to download your FMEA in Excel or CSV format.",
            'action_required': 'navigate_to_export'
        }
    
    def handle_general_query(self, message: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Handle general queries about FMEA"""
        # Provide helpful general response
        response = "I'm here to help you with your FMEA! I can assist you with:\n\n"
        response += "🔍 **Analyzing** your current FMEA data\n"
        response += "✏️ **Modifying** entries (severity, occurrence, detection ratings)\n"
        response += "➕ **Adding** new failure modes and components\n"
        response += "❌ **Deleting** unnecessary entries\n"
        response += "❓ **Explaining** FMEA concepts and ratings\n"
        response += "💡 **Suggesting** improvements and best practices\n"
        response += "✅ **Validating** your FMEA for completeness\n"
        response += "📊 **Exporting** your FMEA to Excel or CSV\n\n"
        
        if not fmea_data.empty:
            response += f"Your current FMEA has {len(fmea_data)} entries. "
            high_risk_count = len(fmea_data[fmea_data['RPN'] > 100]) if 'RPN' in fmea_data.columns else 0
            response += f"I notice {high_risk_count} high-risk items that might need attention.\n\n"
        
        response += "What would you like to work on?"
        
        return {
            'message': response,
            'capabilities': list(self.handlers.keys())
        }
    
    def parse_modification_request(self, message: str, fmea_data: pd.DataFrame) -> Optional[Dict]:
        """Parse modification request from user message"""
        try:
            message_lower = message.lower()
            
            # Extract component/failure mode
            component_match = None
            failure_mode_match = None
            
            for _, row in fmea_data.iterrows():
                component = str(row['Component']).lower()
                failure_mode = str(row['Failure Mode']).lower()
                
                if component in message_lower or any(word in component for word in message_lower.split()):
                    component_match = row
                    break
                elif failure_mode in message_lower or any(word in failure_mode for word in message_lower.split()):
                    failure_mode_match = row
                    break
            
            target_row = component_match or failure_mode_match
            if target_row is None:
                return None
            
            # Extract field to modify and new value
            field_patterns = {
                'severity': r'severity\s*(?:to|=|:)\s*(\d+)',
                'occurrence': r'occurrence\s*(?:to|=|:)\s*(\d+)', 
                'detection': r'detection\s*(?:to|=|:)\s*(\d+)'
            }
            
            for field, pattern in field_patterns.items():
                match = re.search(pattern, message_lower)
                if match:
                    new_value = int(match.group(1))
                    if 1 <= new_value <= 10:
                        row_index = target_row.name
                        return {
                            'type': 'field_update',
                            'row_index': row_index,
                            'field': field.title(),
                            'new_value': new_value,
                            'description': f"Set {field} to {new_value} for {target_row['Component']} - {target_row['Failure Mode']}"
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse modification request: {str(e)}")
            return None
    
    def parse_add_request(self, message: str) -> Optional[Dict]:
        """Parse add request from user message"""
        try:
            message_lower = message.lower()
            entry_info = {}
            
            # Extract component
            component_patterns = [
                r'(?:add|create).*?(?:for|to)\s+([a-zA-Z\s]+?)(?:\s|$)',
                r'component[:\s]+([a-zA-Z\s]+?)(?:\s|$)',
                r'([a-zA-Z\s]+?)\s+(?:failure|component)'
            ]
            
            for pattern in component_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    entry_info['component'] = match.group(1).strip().title()
                    break
            
            # Extract failure mode
            failure_patterns = [
                r'failure[:\s]+([a-zA-Z\s]+?)(?:\s|$)',
                r'mode[:\s]+([a-zA-Z\s]+?)(?:\s|$)',
                r'([a-zA-Z\s]*(?:fail|break|malfunction|error)[a-zA-Z\s]*)'
            ]
            
            for pattern in failure_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    entry_info['failure_mode'] = match.group(1).strip()
                    break
            
            # Extract effect if mentioned
            effect_patterns = [
                r'effect[:\s]+([a-zA-Z\s]+?)(?:\s|$)',
                r'causes?[:\s]+([a-zA-Z\s]+?)(?:\s|$)'
            ]
            
            for pattern in effect_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    entry_info['effect'] = match.group(1).strip()
                    break
            
            return entry_info if entry_info else None
            
        except Exception as e:
            self.logger.error(f"Failed to parse add request: {str(e)}")
            return None
    
    def parse_delete_request(self, message: str, fmea_data: pd.DataFrame) -> Optional[Dict]:
        """Parse delete request from user message"""
        try:
            message_lower = message.lower()
            
            # Check for row number
            number_match = re.search(r'(?:row|entry|number)\s*(\d+)', message_lower)
            if number_match:
                row_num = int(number_match.group(1))
                if 0 <= row_num < len(fmea_data):
                    row = fmea_data.iloc[row_num]
                    return {
                        'index': row_num,
                        'description': f"{row['Component']} - {row['Failure Mode']}"
                    }
            
            # Check for component/failure mode match
            for idx, (_, row) in enumerate(fmea_data.iterrows()):
                component = str(row['Component']).lower()
                failure_mode = str(row['Failure Mode']).lower()
                
                if component in message_lower or failure_mode in message_lower:
                    return {
                        'index': idx,
                        'description': f"{row['Component']} - {row['Failure Mode']}"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse delete request: {str(e)}")
            return None
    
    def identify_explanation_topic(self, message: str) -> str:
        """Identify what topic user wants explained"""
        message_lower = message.lower()
        
        topic_keywords = {
            'severity': ['severity', 'sev', 'impact', 'serious'],
            'occurrence': ['occurrence', 'occ', 'probability', 'likelihood', 'frequency'],
            'detection': ['detection', 'det', 'detect', 'find', 'catch'],
            'rpn': ['rpn', 'risk priority', 'priority number', 'calculation'],
            'fmea': ['fmea', 'failure mode', 'analysis', 'methodology']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def apply_modification(self, fmea_data: pd.DataFrame, modification: Dict) -> pd.DataFrame:
        """Apply modification to FMEA data"""
        try:
            updated_fmea = fmea_data.copy()
            
            if modification['type'] == 'field_update':
                row_idx = modification['row_index']
                field = modification['field']
                new_value = modification['new_value']
                
                updated_fmea.loc[row_idx, field] = new_value
                
                # Recalculate RPN if S, O, or D was changed
                if field in ['Severity', 'Occurrence', 'Detection']:
                    updated_fmea.loc[row_idx, 'RPN'] = (
                        updated_fmea.loc[row_idx, 'Severity'] * 
                        updated_fmea.loc[row_idx, 'Occurrence'] * 
                        updated_fmea.loc[row_idx, 'Detection']
                    )
            
            return updated_fmea
            
        except Exception as e:
            self.logger.error(f"Failed to apply modification: {str(e)}")
            return fmea_data
    
    def create_fmea_entry(self, entry_info: Dict) -> Dict:
        """Create new FMEA entry from provided information"""
        return {
            'Component': entry_info.get('component', 'New Component'),
            'Failure Mode': entry_info.get('failure_mode', 'New Failure Mode'),
            'Effect': entry_info.get('effect', 'To be determined'),
            'Cause': entry_info.get('cause', 'To be determined'),
            'Detection Method': entry_info.get('detection_method', 'To be determined'),
            'Severity': entry_info.get('severity', 5),
            'Occurrence': entry_info.get('occurrence', 5),
            'Detection': entry_info.get('detection', 5),
            'RPN': entry_info.get('severity', 5) * entry_info.get('occurrence', 5) * entry_info.get('detection', 5),
            'Source': 'User Input'
        }
    
    def generate_fmea_analysis(self, fmea_data: pd.DataFrame) -> str:
        """Generate comprehensive FMEA analysis"""
        try:
            analysis = ""
            
            # Basic statistics
            total_entries = len(fmea_data)
            avg_rpn = fmea_data['RPN'].mean() if 'RPN' in fmea_data.columns else 0
            max_rpn = fmea_data['RPN'].max() if 'RPN' in fmea_data.columns else 0
            
            analysis += f"📊 **FMEA Overview:**\n"
            analysis += f"• Total entries: {total_entries}\n"
            analysis += f"• Average RPN: {avg_rpn:.1f}\n"
            analysis += f"• Maximum RPN: {max_rpn}\n\n"
            
            # Risk distribution
            if 'RPN' in fmea_data.columns:
                high_risk = len(fmea_data[fmea_data['RPN'] > 100])
                medium_risk = len(fmea_data[(fmea_data['RPN'] >= 50) & (fmea_data['RPN'] <= 100)])
                low_risk = len(fmea_data[fmea_data['RPN'] < 50])
                
                analysis += f"🎯 **Risk Distribution:**\n"
                analysis += f"• High Risk (RPN > 100): {high_risk} entries\n"
                analysis += f"• Medium Risk (RPN 50-100): {medium_risk} entries\n"
                analysis += f"• Low Risk (RPN < 50): {low_risk} entries\n\n"
                
                # Top risk items
                if high_risk > 0:
                    top_risks = fmea_data.nlargest(3, 'RPN')
                    analysis += f"⚠️ **Top Risk Items:**\n"
                    for idx, (_, row) in enumerate(top_risks.iterrows(), 1):
                        analysis += f"{idx}. {row['Component']} - {row['Failure Mode']} (RPN: {row['RPN']})\n"
                    analysis += "\n"
            
            # Component analysis
            if 'Component' in fmea_data.columns:
                component_counts = fmea_data['Component'].value_counts()
                analysis += f"🔧 **Components with Most Failure Modes:**\n"
                for component, count in component_counts.head(3).items():
                    analysis += f"• {component}: {count} failure modes\n"
            
            return analysis
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def generate_suggestions(self, fmea_data: pd.DataFrame) -> str:
        """Generate improvement suggestions"""
        try:
            suggestions = ""
            
            if 'RPN' in fmea_data.columns:
                high_rpn_items = fmea_data[fmea_data['RPN'] > 100]
                
                if len(high_rpn_items) > 0:
                    suggestions += "🔴 **Immediate Actions Needed:**\n"
                    for _, row in high_rpn_items.head(3).iterrows():
                        suggestions += f"• **{row['Component']}**: "
                        if row['Severity'] >= 8:
                            suggestions += "Consider design changes to reduce impact severity. "
                        if row['Occurrence'] >= 7:
                            suggestions += "Implement preventive measures. "
                        if row['Detection'] >= 7:
                            suggestions += "Improve detection methods. "
                        suggestions += f"(Current RPN: {row['RPN']})\n"
                    suggestions += "\n"
                
                # General suggestions
                avg_severity = fmea_data['Severity'].mean() if 'Severity' in fmea_data.columns else 0
                avg_occurrence = fmea_data['Occurrence'].mean() if 'Occurrence' in fmea_data.columns else 0
                avg_detection = fmea_data['Detection'].mean() if 'Detection' in fmea_data.columns else 0
                
                suggestions += "💡 **General Recommendations:**\n"
                
                if avg_occurrence > 6:
                    suggestions += "• Focus on preventive maintenance to reduce failure occurrence\n"
                if avg_detection > 6:
                    suggestions += "• Invest in better monitoring and detection systems\n"
                if avg_severity > 7:
                    suggestions += "• Consider design modifications to reduce failure impact\n"
                
                suggestions += "• Regular FMEA reviews and updates\n"
                suggestions += "• Training for operators on failure recognition\n"
                suggestions += "• Establish clear escalation procedures\n"
            
            return suggestions
            
        except Exception as e:
            return f"Suggestion generation failed: {str(e)}"
    
    def validate_fmea_data(self, fmea_data: pd.DataFrame) -> str:
        """Validate FMEA data and provide feedback"""
        try:
            validation = ""
            issues = []
            
            # Check required columns
            required_cols = ['Component', 'Failure Mode', 'Effect', 'Cause']
            missing_cols = [col for col in required_cols if col not in fmea_data.columns]
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            
            # Check for empty values
            for col in required_cols:
                if col in fmea_data.columns:
                    empty_count = fmea_data[col].isna().sum() + (fmea_data[col] == '').sum()
                    if empty_count > 0:
                        issues.append(f"{empty_count} empty values in {col}")
            
            # Check rating ranges
            rating_cols = ['Severity', 'Occurrence', 'Detection']
            for col in rating_cols:
                if col in fmea_data.columns:
                    invalid = ((fmea_data[col] < 1) | (fmea_data[col] > 10)).sum()
                    if invalid > 0:
                        issues.append(f"{invalid} invalid {col} ratings (must be 1-10)")
            
            if issues:
                validation = "❌ **Validation Issues Found:**\n"
                for issue in issues:
                    validation += f"• {issue}\n"
            else:
                validation = "✅ **Validation Passed:**\n"
                validation += "• All required fields are complete\n"
                validation += "• All ratings are within valid ranges\n"
                validation += "• FMEA structure is correct\n"
            
            return validation
            
        except Exception as e:
            return f"Validation failed: {str(e)}"
    
    def explain_severity(self) -> str:
        """Explain severity rating scale"""
        explanation = "**Severity Rating (1-10)** measures the impact of a failure on the customer or system:\n\n"
        
        for rating, description in self.fmea_knowledge['severity_scale'].items():
            explanation += f"**{rating}**: {description}\n"
        
        explanation += "\n💡 **Tips for Rating Severity:**\n"
        explanation += "• Consider the worst-case scenario\n"
        explanation += "• Focus on customer impact, not internal issues\n"
        explanation += "• Safety issues should always be rated 9-10\n"
        explanation += "• Severity rarely changes unless design is modified\n"
        
        return explanation
    
    def explain_occurrence(self) -> str:
        """Explain occurrence rating scale"""
        explanation = "**Occurrence Rating (1-10)** measures how likely a failure is to happen:\n\n"
        
        for rating, description in self.fmea_knowledge['occurrence_scale'].items():
            explanation += f"**{rating}**: {description}\n"
        
        explanation += "\n💡 **Tips for Rating Occurrence:**\n"
        explanation += "• Use historical data when available\n"
        explanation += "• Consider operating conditions and environment\n"
        explanation += "• Preventive maintenance can reduce occurrence\n"
        explanation += "• New designs may have higher uncertainty\n"
        
        return explanation
    
    def explain_detection(self) -> str:
        """Explain detection rating scale"""
        explanation = "**Detection Rating (1-10)** measures the probability of detecting a failure before it reaches the customer:\n\n"
        
        for rating, description in self.fmea_knowledge['detection_scale'].items():
            explanation += f"**{rating}**: {description}\n"
        
        explanation += "\n💡 **Tips for Rating Detection:**\n"
        explanation += "• Consider current inspection and testing methods\n"
        explanation += "• Automated detection systems typically rate lower (better)\n"
        explanation += "• Visual inspection alone typically rates higher (worse)\n"
        explanation += "• Detection can be improved with better monitoring\n"
        
        return explanation
    
    def explain_rpn(self) -> str:
        """Explain RPN calculation and interpretation"""
        explanation = "**Risk Priority Number (RPN)** is calculated as:\n\n"
        explanation += "**RPN = Severity × Occurrence × Detection**\n\n"
        explanation += "**RPN Ranges:**\n"
        explanation += "• **1-50**: Low Risk - Monitor and review\n"
        explanation += "• **51-100**: Medium Risk - Consider action\n"
        explanation += "• **101-200**: High Risk - Action recommended\n"
        explanation += "• **201-1000**: Critical Risk - Immediate action required\n\n"
        explanation += "💡 **RPN Guidelines:**\n"
        explanation += "• Focus on highest RPN values first\n"
        explanation += "• Consider individual S, O, D values too\n"
        explanation += "• Any severity 9-10 should be addressed regardless of RPN\n"
        explanation += "• Target RPN reduction through prevention and detection\n"
        
        return explanation
    
    def explain_fmea_general(self) -> str:
        """Explain general FMEA methodology"""
        explanation = "**Failure Mode and Effects Analysis (FMEA)** is a systematic method for:\n\n"
        explanation += "🔍 **Identifying** potential failure modes\n"
        explanation += "📊 **Assessing** their impact and likelihood\n"
        explanation += "🎯 **Prioritizing** risks for action\n"
        explanation += "🛠️ **Preventing** failures through design or process improvements\n\n"
        explanation += "**FMEA Process:**\n"
        explanation += "1. Identify system components and functions\n"
        explanation += "2. Determine potential failure modes\n"
        explanation += "3. Analyze effects of each failure\n"
        explanation += "4. Identify potential causes\n"
        explanation += "5. Rate Severity, Occurrence, and Detection\n"
        explanation += "6. Calculate RPN and prioritize actions\n"
        explanation += "7. Implement improvements and re-evaluate\n\n"
        explanation += "**Benefits:**\n"
        explanation += "• Proactive risk management\n"
        explanation += "• Improved product reliability\n"
        explanation += "• Reduced warranty costs\n"
        explanation += "• Enhanced customer satisfaction\n"
        
        return explanation
    
    def accept_corrections(self, feedback: str, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Accept user corrections and feedback"""
        return {
            'message': "Thank you for the feedback! I'll use this to improve future suggestions.",
            'feedback_recorded': feedback,
            'timestamp': datetime.now()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': 'ChatAgent',
            'conversation_length': len(self.conversation_history),
            'available_handlers': list(self.handlers.keys()),
            'knowledge_loaded': bool(self.fmea_knowledge),
            'active': True
        }