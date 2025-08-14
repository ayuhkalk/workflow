# agents/fmea_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
import re
from datetime import datetime

class FMEAAgent:
    """
    Generates and manages FMEA tables using extracted document data
    Implements the FMEAAgent from the class diagram
    """
    
    def __init__(self):
        self.fmea_generator = FMEAGenerator()
        self.logger = logging.getLogger(__name__)
        
        # Load FMEA templates and knowledge base
        self.load_fmea_knowledge_base()
    
    def load_fmea_knowledge_base(self):
        """Load predefined FMEA knowledge for common machinery components"""
        self.knowledge_base = {
            'motor': {
                'failure_modes': [
                    'Motor overheating', 'Motor failure', 'Electrical connection failure',
                    'Motor bearing failure', 'Motor winding failure'
                ],
                'effects': [
                    'Loss of motion', 'System shutdown', 'Reduced performance',
                    'Noise and vibration', 'Fire hazard'
                ],
                'causes': [
                    'Overload', 'Poor ventilation', 'Electrical surge',
                    'Bearing wear', 'Contamination'
                ],
                'detection_methods': [
                    'Temperature monitoring', 'Vibration analysis', 'Current monitoring',
                    'Visual inspection', 'Preventive maintenance'
                ]
            },
            'bearing': {
                'failure_modes': [
                    'Bearing seizure', 'Bearing wear', 'Bearing contamination',
                    'Bearing misalignment', 'Lubrication failure'
                ],
                'effects': [
                    'Machine stoppage', 'Increased vibration', 'Noise generation',
                    'Heat generation', 'Premature component failure'
                ],
                'causes': [
                    'Inadequate lubrication', 'Contamination', 'Overload',
                    'Misalignment', 'Normal wear'
                ],
                'detection_methods': [
                    'Vibration monitoring', 'Temperature monitoring', 'Oil analysis',
                    'Visual inspection', 'Acoustic monitoring'
                ]
            },
            'switch': {
                'failure_modes': [
                    'Switch failure to operate', 'False switch activation',
                    'Switch contact degradation', 'Switch mechanical failure'
                ],
                'effects': [
                    'Loss of control', 'Safety system failure', 'False alarms',
                    'System malfunction'
                ],
                'causes': [
                    'Contact wear', 'Environmental contamination', 'Mechanical damage',
                    'Electrical overload', 'Age deterioration'
                ],
                'detection_methods': [
                    'Functional testing', 'Visual inspection', 'Electrical testing',
                    'Preventive replacement'
                ]
            },
            'axis_drive': {
                'failure_modes': [
                    'Drive overheating', 'Drive electronics failure', 'Motor connection failure',
                    'Position feedback failure', 'Current regulation failure'
                ],
                'effects': [
                    'Axis positioning error', 'Machine shutdown', 'Reduced accuracy',
                    'Safety system activation', 'Production delays'
                ],
                'causes': [
                    'Electrical component failure', 'Overload conditions', 'Environmental factors',
                    'Wiring issues', 'Software errors'
                ],
                'detection_methods': [
                    'Position feedback monitoring', 'Current monitoring', 'Temperature monitoring',
                    'Diagnostic software', 'Alarm systems'
                ]
            },
            'lubrication_system': {
                'failure_modes': [
                    'Lubrication pump failure', 'Oil contamination', 'Insufficient lubrication',
                    'Oil line blockage', 'Oil leak'
                ],
                'effects': [
                    'Component wear', 'Overheating', 'Premature failure',
                    'Increased maintenance', 'System reliability issues'
                ],
                'causes': [
                    'Pump wear', 'Contamination', 'Filter blockage',
                    'Line damage', 'Seal failure'
                ],
                'detection_methods': [
                    'Pressure monitoring', 'Oil analysis', 'Visual inspection',
                    'Flow monitoring', 'Scheduled maintenance'
                ]
            }
        }
    
    def generate_initial_fmea(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate initial FMEA table from extracted data"""
        try:
            self.logger.info("Generating initial FMEA table")
            
            # Extract relevant information from context
            extracted_text = context.get('extracted_text', '')
            failure_modes = context.get('failure_modes', [])
            technical_specs = context.get('technical_specs', {})
            historical_fmea = context.get('historical_fmea')
            
            # Identify components mentioned in the text
            identified_components = self._identify_components(extracted_text, technical_specs)
            
            # Generate FMEA entries
            fmea_entries = []
            
            # Generate from identified components
            for component in identified_components:
                component_entries = self._generate_component_fmea(component, extracted_text)
                fmea_entries.extend(component_entries)
            
            # Generate from extracted failure modes
            for failure_mode_data in failure_modes:
                entry = self._generate_fmea_from_failure_mode(failure_mode_data, extracted_text)
                if entry:
                    fmea_entries.append(entry)
            
            # If we have historical FMEA, use it as reference
            if historical_fmea is not None and not historical_fmea.empty:
                historical_entries = self._adapt_historical_fmea(historical_fmea, identified_components)
                fmea_entries.extend(historical_entries)
            
            # Convert to DataFrame
            if fmea_entries:
                fmea_df = pd.DataFrame(fmea_entries)
                fmea_df = self._standardize_fmea_dataframe(fmea_df)
            else:
                # Create empty FMEA with standard columns
                fmea_df = self._create_empty_fmea()
            
            self.logger.info(f"Generated FMEA with {len(fmea_df)} entries")
            return fmea_df
            
        except Exception as e:
            self.logger.error(f"FMEA generation failed: {str(e)}")
            return self._create_empty_fmea()
    
    def _identify_components(self, text: str, technical_specs: Dict) -> List[str]:
        """Identify components mentioned in the text"""
        components = []
        
        # Check against knowledge base
        for component_type in self.knowledge_base.keys():
            if component_type.lower() in text.lower():
                components.append(component_type)
        
        # Extract from technical specs
        if 'components' in technical_specs:
            components.extend(technical_specs['components'])
        
        # Common CNC machine components
        cnc_components = [
            'spindle', 'motor', 'bearing', 'belt', 'switch', 'drive', 'axis',
            'lubrication', 'coolant', 'table', 'head', 'control', 'fuse'
        ]
        
        for component in cnc_components:
            if component.lower() in text.lower():
                components.append(component)
        
        # Remove duplicates and return
        return list(set(components))
    
    def _generate_component_fmea(self, component: str, context_text: str) -> List[Dict]:
        """Generate FMEA entries for a specific component"""
        entries = []
        
        # Get component data from knowledge base
        component_lower = component.lower()
        kb_data = None
        
        for kb_component, data in self.knowledge_base.items():
            if kb_component in component_lower or component_lower in kb_component:
                kb_data = data
                break
        
        if kb_data:
            # Generate entries based on knowledge base
            for i, failure_mode in enumerate(kb_data['failure_modes'][:3]):  # Limit to 3 per component
                entry = {
                    'Component': component.title(),
                    'Failure Mode': failure_mode,
                    'Effect': kb_data['effects'][i % len(kb_data['effects'])],
                    'Cause': kb_data['causes'][i % len(kb_data['causes'])],
                    'Detection Method': kb_data['detection_methods'][i % len(kb_data['detection_methods'])],
                    'Severity': 7,  # Default values - user should adjust
                    'Occurrence': 4,
                    'Detection': 5,
                    'RPN': 7 * 4 * 5,
                    'Source': 'Knowledge Base'
                }
                entries.append(entry)
        
        return entries
    
    def _generate_fmea_from_failure_mode(self, failure_mode_data: Dict, context_text: str) -> Optional[Dict]:
        """Generate FMEA entry from extracted failure mode"""
        try:
            failure_mode = failure_mode_data.get('failure_mode', '')
            
            # Try to infer component, effect, and cause from context
            component = self._infer_component(failure_mode, context_text)
            effect = self._infer_effect(failure_mode)
            cause = self._infer_cause(failure_mode)
            detection = self._infer_detection_method(failure_mode)
            
            return {
                'Component': component,
                'Failure Mode': failure_mode,
                'Effect': effect,
                'Cause': cause,
                'Detection Method': detection,
                'Severity': 6,  # Default values
                'Occurrence': 4,
                'Detection': 6,
                'RPN': 6 * 4 * 6,
                'Source': 'Document Extraction'
            }
        except Exception as e:
            self.logger.error(f"Failed to generate FMEA from failure mode: {str(e)}")
            return None
    
    def _infer_component(self, failure_mode: str, context: str) -> str:
        """Infer component from failure mode and context"""
        failure_mode_lower = failure_mode.lower()
        
        # Check for component keywords in failure mode
        for component in self.knowledge_base.keys():
            if component in failure_mode_lower:
                return component.title()
        
        # Check context around the failure mode
        common_components = ['motor', 'bearing', 'switch', 'drive', 'spindle', 'axis']
        for component in common_components:
            if component in failure_mode_lower:
                return component.title()
        
        return "Unknown Component"
    
    def _infer_effect(self, failure_mode: str) -> str:
        """Infer effect from failure mode"""
        failure_lower = failure_mode.lower()
        
        if any(word in failure_lower for word in ['fail', 'stop', 'break']):
            return "System malfunction or stoppage"
        elif any(word in failure_lower for word in ['wear', 'degradation']):
            return "Reduced performance and reliability"
        elif any(word in failure_lower for word in ['leak', 'contamination']):
            return "Environmental contamination and component damage"
        else:
            return "Potential system impact - requires analysis"
    
    def _infer_cause(self, failure_mode: str) -> str:
        """Infer potential cause from failure mode"""
        failure_lower = failure_mode.lower()
        
        if any(word in failure_lower for word in ['wear', 'aging']):
            return "Normal wear and aging"
        elif any(word in failure_lower for word in ['contamination', 'dirt']):
            return "Environmental contamination"
        elif any(word in failure_lower for word in ['overload', 'stress']):
            return "Excessive operational stress"
        else:
            return "Multiple potential causes - requires investigation"
    
    def _infer_detection_method(self, failure_mode: str) -> str:
        """Infer detection method from failure mode"""
        failure_lower = failure_mode.lower()
        
        if any(word in failure_lower for word in ['electrical', 'electronic']):
            return "Electrical testing and monitoring"
        elif any(word in failure_lower for word in ['mechanical', 'wear']):
            return "Visual inspection and vibration monitoring"
        elif any(word in failure_lower for word in ['temperature', 'heat']):
            return "Temperature monitoring"
        else:
            return "Regular inspection and preventive maintenance"
    
    def _adapt_historical_fmea(self, historical_fmea: pd.DataFrame, current_components: List[str]) -> List[Dict]:
        """Adapt historical FMEA data to current context"""
        entries = []
        
        try:
            # Standardize historical FMEA column names
            column_mapping = self._map_historical_columns(historical_fmea.columns)
            
            for _, row in historical_fmea.iterrows():
                # Check if this historical entry is relevant to current components
                component = str(row.get(column_mapping.get('component', ''), 'Unknown'))
                
                if any(comp.lower() in component.lower() for comp in current_components):
                    entry = {
                        'Component': component,
                        'Failure Mode': str(row.get(column_mapping.get('failure_mode', ''), '')),
                        'Effect': str(row.get(column_mapping.get('effect', ''), '')),
                        'Cause': str(row.get(column_mapping.get('cause', ''), '')),
                        'Detection Method': str(row.get(column_mapping.get('detection', ''), '')),
                        'Severity': self._safe_int_convert(row.get(column_mapping.get('severity', ''), 5)),
                        'Occurrence': self._safe_int_convert(row.get(column_mapping.get('occurrence', ''), 3)),
                        'Detection': self._safe_int_convert(row.get(column_mapping.get('detection_rating', ''), 4)),
                        'Source': 'Historical Data'
                    }
                    entry['RPN'] = entry['Severity'] * entry['Occurrence'] * entry['Detection']
                    entries.append(entry)
        
        except Exception as e:
            self.logger.error(f"Failed to adapt historical FMEA: {str(e)}")
        
        return entries
    
    def _map_historical_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map historical FMEA columns to standard format"""
        column_mapping = {}
        columns_lower = [col.lower().strip() for col in columns]
        
        mappings = {
            'component': ['component', 'system', 'item', 'part'],
            'failure_mode': ['failure mode', 'failure', 'mode'],
            'effect': ['effect', 'failure effect', 'consequence'],
            'cause': ['cause', 'failure cause', 'root cause'],
            'severity': ['severity', 'sev', 's'],
            'occurrence': ['occurrence', 'occ', 'o'],
            'detection': ['detection method', 'detection', 'det', 'd'],
            'detection_rating': ['detection rating', 'detect', 'detectability']
        }
        
        for standard_col, possible_names in mappings.items():
            for possible_name in possible_names:
                for i, col in enumerate(columns_lower):
                    if possible_name in col:
                        column_mapping[standard_col] = columns[i]
                        break
                if standard_col in column_mapping:
                    break
        
        return column_mapping
    
    def _safe_int_convert(self, value, default: int = 5) -> int:
        """Safely convert value to integer"""
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def _standardize_fmea_dataframe(self, fmea_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize FMEA dataframe format"""
        standard_columns = [
            'Component', 'Failure Mode', 'Effect', 'Cause', 'Detection Method',
            'Severity', 'Occurrence', 'Detection', 'RPN', 'Source'
        ]
        
        # Ensure all standard columns exist
        for col in standard_columns:
            if col not in fmea_df.columns:
                if col in ['Severity', 'Occurrence', 'Detection']:
                    fmea_df[col] = 5  # Default rating
                elif col == 'RPN':
                    fmea_df[col] = fmea_df.get('Severity', 5) * fmea_df.get('Occurrence', 5) * fmea_df.get('Detection', 5)
                else:
                    fmea_df[col] = 'To be determined'
        
        # Reorder columns
        fmea_df = fmea_df[standard_columns]
        
        # Calculate RPN if not already calculated
        fmea_df['RPN'] = fmea_df['Severity'] * fmea_df['Occurrence'] * fmea_df['Detection']
        
        return fmea_df
    
    def _create_empty_fmea(self) -> pd.DataFrame:
        """Create empty FMEA dataframe with standard columns"""
        columns = [
            'Component', 'Failure Mode', 'Effect', 'Cause', 'Detection Method',
            'Severity', 'Occurrence', 'Detection', 'RPN', 'Source'
        ]
        return pd.DataFrame(columns=columns)
    
    def improve_with_historical_data(self, current_fmea: pd.DataFrame, historical_fmea: pd.DataFrame) -> pd.DataFrame:
        """Improve current FMEA using historical data"""
        try:
            # Update severity, occurrence, and detection ratings based on historical data
            for idx, row in current_fmea.iterrows():
                historical_match = self._find_historical_match(row, historical_fmea)
                if historical_match is not None:
                    current_fmea.loc[idx, 'Severity'] = historical_match.get('Severity', row['Severity'])
                    current_fmea.loc[idx, 'Occurrence'] = historical_match.get('Occurrence', row['Occurrence'])
                    current_fmea.loc[idx, 'Detection'] = historical_match.get('Detection', row['Detection'])
            
            # Recalculate RPN
            current_fmea['RPN'] = current_fmea['Severity'] * current_fmea['Occurrence'] * current_fmea['Detection']
            
            return current_fmea
        
        except Exception as e:
            self.logger.error(f"Failed to improve with historical data: {str(e)}")
            return current_fmea
    
    def _find_historical_match(self, current_row: pd.Series, historical_fmea: pd.DataFrame) -> Optional[Dict]:
        """Find matching entry in historical FMEA"""
        try:
            # Look for similar failure modes or components
            current_failure_mode = str(current_row.get('Failure Mode', '')).lower()
            current_component = str(current_row.get('Component', '')).lower()
            
            for _, hist_row in historical_fmea.iterrows():
                hist_failure_mode = str(hist_row.get('Failure Mode', '')).lower()
                hist_component = str(hist_row.get('Component', '')).lower()
                
                # Check for similarity
                if (current_failure_mode in hist_failure_mode or hist_failure_mode in current_failure_mode or
                    current_component in hist_component or hist_component in current_component):
                    return hist_row.to_dict()
            
            return None
        except Exception:
            return None
    
    def analyze_high_risk_items(self, fmea_data: pd.DataFrame) -> str:
        """Analyze high-risk items in FMEA"""
        try:
            if fmea_data.empty or 'RPN' not in fmea_data.columns:
                return "No FMEA data available for analysis."
            
            # Sort by RPN
            sorted_fmea = fmea_data.sort_values('RPN', ascending=False)
            high_risk_threshold = 100
            high_risk_items = sorted_fmea[sorted_fmea['RPN'] > high_risk_threshold]
            
            analysis = f"High Risk Analysis Results:\n\n"
            analysis += f"Total FMEA entries: {len(fmea_data)}\n"
            analysis += f"High risk items (RPN > {high_risk_threshold}): {len(high_risk_items)}\n"
            analysis += f"Average RPN: {fmea_data['RPN'].mean():.1f}\n"
            analysis += f"Maximum RPN: {fmea_data['RPN'].max()}\n\n"
            
            if len(high_risk_items) > 0:
                analysis += "Top 5 Highest Risk Items:\n"
                for idx, (_, row) in enumerate(high_risk_items.head().iterrows()):
                    analysis += f"{idx+1}. {row['Component']} - {row['Failure Mode']} (RPN: {row['RPN']})\n"
                    analysis += f"   Effect: {row['Effect']}\n"
                    analysis += f"   S={row['Severity']}, O={row['Occurrence']}, D={row['Detection']}\n\n"
            else:
                analysis += "No high-risk items identified based on current threshold.\n"
            
            return analysis
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def suggest_improvements(self, fmea_data: pd.DataFrame) -> str:
        """Suggest improvements for FMEA"""
        try:
            if fmea_data.empty:
                return "No FMEA data available for improvement suggestions."
            
            suggestions = "FMEA Improvement Suggestions:\n\n"
            
            # Analyze RPN distribution
            high_rpn = fmea_data[fmea_data['RPN'] > 100]
            medium_rpn = fmea_data[(fmea_data['RPN'] >= 50) & (fmea_data['RPN'] <= 100)]
            
            if len(high_rpn) > 0:
                suggestions += "🔴 HIGH PRIORITY ACTIONS:\n"
                for _, row in high_rpn.iterrows():
                    suggestions += f"• {row['Component']} - {row['Failure Mode']}: "
                    
                    # Suggest specific improvements based on S-O-D values
                    if row['Severity'] >= 8:
                        suggestions += "Consider design changes to reduce severity. "
                    if row['Occurrence'] >= 7:
                        suggestions += "Implement preventive measures to reduce occurrence. "
                    if row['Detection'] >= 7:
                        suggestions += "Improve detection methods and monitoring. "
                    
                    suggestions += f"(Current RPN: {row['RPN']})\n"
                suggestions += "\n"
            
            if len(medium_rpn) > 0:
                suggestions += "🟡 MEDIUM PRIORITY ACTIONS:\n"
                suggestions += f"• Review {len(medium_rpn)} medium-risk items for potential improvements\n"
                suggestions += "• Consider implementing condition monitoring\n"
                suggestions += "• Update maintenance schedules\n\n"
            
            # General suggestions
            suggestions += "📋 GENERAL RECOMMENDATIONS:\n"
            suggestions += "• Regularly review and update RPN ratings\n"
            suggestions += "• Implement preventive maintenance programs\n"
            suggestions += "• Consider adding redundancy for critical components\n"
            suggestions += "• Train operators on early failure detection\n"
            suggestions += "• Establish clear escalation procedures\n"
            
            return suggestions
            
        except Exception as e:
            return f"Suggestion generation failed: {str(e)}"
    
    def validate_fmea(self, fmea_data: pd.DataFrame) -> str:
        """Validate FMEA completeness and consistency"""
        try:
            if fmea_data.empty:
                return "❌ No FMEA data to validate."
            
            validation_results = "FMEA Validation Results:\n\n"
            issues = []
            
            # Check for required columns
            required_columns = ['Component', 'Failure Mode', 'Effect', 'Cause', 'Detection Method']
            missing_columns = [col for col in required_columns if col not in fmea_data.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            # Check for empty cells
            for col in required_columns:
                if col in fmea_data.columns:
                    empty_count = fmea_data[col].isna().sum() + (fmea_data[col] == '').sum()
                    if empty_count > 0:
                        issues.append(f"{empty_count} empty entries in '{col}' column")
            
            # Check RPN ratings
            if all(col in fmea_data.columns for col in ['Severity', 'Occurrence', 'Detection']):
                invalid_severity = ((fmea_data['Severity'] < 1) | (fmea_data['Severity'] > 10)).sum()
                invalid_occurrence = ((fmea_data['Occurrence'] < 1) | (fmea_data['Occurrence'] > 10)).sum()
                invalid_detection = ((fmea_data['Detection'] < 1) | (fmea_data['Detection'] > 10)).sum()
                
                if invalid_severity > 0:
                    issues.append(f"{invalid_severity} invalid severity ratings (must be 1-10)")
                if invalid_occurrence > 0:
                    issues.append(f"{invalid_occurrence} invalid occurrence ratings (must be 1-10)")
                if invalid_detection > 0:
                    issues.append(f"{invalid_detection} invalid detection ratings (must be 1-10)")
            
            # Check for duplicate entries
            if len(required_columns) >= 2:
                duplicates = fmea_data.duplicated(subset=required_columns[:2]).sum()
                if duplicates > 0:
                    issues.append(f"{duplicates} potential duplicate entries found")
            
            # Generate validation summary
            if issues:
                validation_results += "❌ VALIDATION ISSUES FOUND:\n"
                for i, issue in enumerate(issues, 1):
                    validation_results += f"{i}. {issue}\n"
                validation_results += f"\nTotal issues: {len(issues)}\n"
            else:
                validation_results += "✅ VALIDATION PASSED:\n"
                validation_results += "• All required columns present\n"
                validation_results += "• No empty critical fields\n"
                validation_results += "• RPN ratings within valid range\n"
                validation_results += "• No duplicate entries detected\n"
            
            # Add statistics
            validation_results += f"\n📊 FMEA STATISTICS:\n"
            validation_results += f"• Total entries: {len(fmea_data)}\n"
            if 'RPN' in fmea_data.columns:
                validation_results += f"• Average RPN: {fmea_data['RPN'].mean():.1f}\n"
                validation_results += f"• High risk items (RPN > 100): {(fmea_data['RPN'] > 100).sum()}\n"
            
            return validation_results
            
        except Exception as e:
            return f"Validation failed: {str(e)}"
    
    def apply_improvements(self, fmea_data: pd.DataFrame, improvements: Dict) -> pd.DataFrame:
        """Apply improvement suggestions to FMEA"""
        try:
            improved_fmea = fmea_data.copy()
            
            for improvement in improvements.get('suggestions', []):
                # Apply specific improvements based on type
                if improvement['type'] == 'rpn_adjustment':
                    row_idx = improvement['row_index']
                    if row_idx < len(improved_fmea):
                        improved_fmea.loc[row_idx, 'Severity'] = improvement.get('new_severity', improved_fmea.loc[row_idx, 'Severity'])
                        improved_fmea.loc[row_idx, 'Occurrence'] = improvement.get('new_occurrence', improved_fmea.loc[row_idx, 'Occurrence'])
                        improved_fmea.loc[row_idx, 'Detection'] = improvement.get('new_detection', improved_fmea.loc[row_idx, 'Detection'])
                
                elif improvement['type'] == 'detection_method_update':
                    component = improvement['component']
                    new_method = improvement['new_method']
                    mask = improved_fmea['Component'] == component
                    improved_fmea.loc[mask, 'Detection Method'] = new_method
            
            # Recalculate RPN
            if all(col in improved_fmea.columns for col in ['Severity', 'Occurrence', 'Detection']):
                improved_fmea['RPN'] = improved_fmea['Severity'] * improved_fmea['Occurrence'] * improved_fmea['Detection']
            
            return improved_fmea
            
        except Exception as e:
            self.logger.error(f"Failed to apply improvements: {str(e)}")
            return fmea_data
    
    def create_fmea_table(self, data: Dict) -> Dict[str, Any]:
        """Create FMEA table - interface method for coordinator"""
        try:
            fmea_df = self.generate_initial_fmea(data)
            return {
                'success': True,
                'fmea_dataframe': fmea_df,
                'entry_count': len(fmea_df)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': 'FMEAAgent',
            'knowledge_base_loaded': bool(self.knowledge_base),
            'component_types': len(self.knowledge_base),
            'active': True
        }


class FMEAGenerator:
    """Specialized FMEA generation logic"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def suggest_risks(self, component: str, context: str) -> List[Dict[str, Any]]:
        """Suggest potential risks for a component based on context"""
        risks = []
        
        # Common risk patterns
        risk_patterns = {
            'wear': {'severity': 6, 'occurrence': 5, 'detection': 4},
            'failure': {'severity': 8, 'occurrence': 3, 'detection': 6},
            'malfunction': {'severity': 7, 'occurrence': 4, 'detection': 5},
            'contamination': {'severity': 5, 'occurrence': 6, 'detection': 3},
            'overheating': {'severity': 8, 'occurrence': 4, 'detection': 4}
        }
        
        for risk_type, ratings in risk_patterns.items():
            if risk_type in context.lower():
                risks.append({
                    'risk_type': risk_type,
                    'component': component,
                    'suggested_ratings': ratings,
                    'confidence': 0.7
                })
        
        return risks
    
    def calculate_rpn(self, severity: int, occurrence: int, detection: int) -> int:
        """Calculate Risk Priority Number"""
        return severity * occurrence * detection
    
    def prioritize_failures(self, fmea_data: pd.DataFrame) -> pd.DataFrame:
        """Prioritize failures by RPN and criticality"""
        if 'RPN' not in fmea_data.columns:
            return fmea_data
        
        # Sort by RPN descending
        prioritized = fmea_data.sort_values('RPN', ascending=False).copy()
        
        # Add priority ranking
        prioritized['Priority_Rank'] = range(1, len(prioritized) + 1)
        
        # Add risk category
        prioritized['Risk_Category'] = pd.cut(
            prioritized['RPN'],
            bins=[0, 50, 100, 200, 1000],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return prioritized