# agents/learning_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path

class LearningAgent:
    """
    Stores and learns from user feedback to improve FMEA suggestions
    Implements the LearningAgent from the class diagram
    """
    
    def __init__(self, feedback_file: str = "fmea_feedback.json"):
        self.logger = logging.getLogger(__name__)
        self.feedback_file = Path(feedback_file)
        self.feedback_data = []
        self.patterns = {}
        
        # Load existing feedback data
        self.load_feedback_data()
        
        # Initialize learning patterns
        self.initialize_learning_patterns()
    
    def load_feedback_data(self):
        """Load existing feedback data from file"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                self.logger.info(f"Loaded {len(self.feedback_data)} feedback records")
            else:
                self.feedback_data = []
                self.logger.info("No existing feedback data found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load feedback data: {str(e)}")
            self.feedback_data = []
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2, default=str)
            self.logger.info(f"Saved {len(self.feedback_data)} feedback records")
        except Exception as e:
            self.logger.error(f"Failed to save feedback data: {str(e)}")
    
    def initialize_learning_patterns(self):
        """Initialize patterns for learning from feedback"""
        self.patterns = {
            'severity_adjustments': {},
            'occurrence_adjustments': {},
            'detection_adjustments': {},
            'component_preferences': {},
            'failure_mode_mappings': {},
            'user_preferences': {}
        }
    
    def store_feedback(self, user_input: str, original_fmea: pd.DataFrame, 
                      modified_fmea: pd.DataFrame) -> Dict[str, Any]:
        """Store user feedback for learning"""
        try:
            feedback_record = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'changes': self._extract_changes(original_fmea, modified_fmea),
                'context': self._extract_context(original_fmea),
                'feedback_type': self._classify_feedback_type(user_input)
            }
            
            self.feedback_data.append(feedback_record)
            
            # Update learning patterns
            self._update_patterns(feedback_record)
            
            # Save to file
            self.save_feedback_data()
            
            self.logger.info(f"Stored feedback: {feedback_record['feedback_type']}")
            
            return {
                'success': True,
                'feedback_id': len(self.feedback_data) - 1,
                'patterns_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to store feedback: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_changes(self, original: pd.DataFrame, modified: pd.DataFrame) -> List[Dict]:
        """Extract changes between original and modified FMEA"""
        changes = []
        
        try:
            # Compare dataframes
            if len(original) != len(modified):
                # Rows added or removed
                if len(modified) > len(original):
                    changes.append({
                        'type': 'rows_added',
                        'count': len(modified) - len(original)
                    })
                else:
                    changes.append({
                        'type': 'rows_removed', 
                        'count': len(original) - len(modified)
                    })
            
            # Compare common rows for value changes
            min_len = min(len(original), len(modified))
            for i in range(min_len):
                row_changes = []
                for col in original.columns:
                    if col in modified.columns:
                        orig_val = original.iloc[i][col]
                        mod_val = modified.iloc[i][col]
                        
                        if orig_val != mod_val:
                            row_changes.append({
                                'column': col,
                                'original_value': orig_val,
                                'modified_value': mod_val
                            })
                
                if row_changes:
                    changes.append({
                        'type': 'value_changes',
                        'row_index': i,
                        'changes': row_changes
                    })
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Failed to extract changes: {str(e)}")
            return []
    
    def _extract_context(self, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract context information from FMEA data"""
        try:
            context = {
                'total_entries': len(fmea_data),
                'components': fmea_data['Component'].unique().tolist() if 'Component' in fmea_data.columns else [],
                'avg_rpn': fmea_data['RPN'].mean() if 'RPN' in fmea_data.columns else None,
                'high_risk_count': len(fmea_data[fmea_data['RPN'] > 100]) if 'RPN' in fmea_data.columns else 0
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to extract context: {str(e)}")
            return {}
    
    def _classify_feedback_type(self, user_input: str) -> str:
        """Classify the type of user feedback"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['severity', 'sev']):
            return 'severity_adjustment'
        elif any(word in user_input_lower for word in ['occurrence', 'occ', 'probability']):
            return 'occurrence_adjustment'
        elif any(word in user_input_lower for word in ['detection', 'det']):
            return 'detection_adjustment'
        elif any(word in user_input_lower for word in ['add', 'create', 'new']):
            return 'entry_addition'
        elif any(word in user_input_lower for word in ['delete', 'remove']):
            return 'entry_removal'
        elif any(word in user_input_lower for word in ['change', 'modify', 'update']):
            return 'entry_modification'
        else:
            return 'general_feedback'
    
    def _update_patterns(self, feedback_record: Dict):
        """Update learning patterns based on feedback"""
        try:
            feedback_type = feedback_record['feedback_type']
            changes = feedback_record['changes']
            
            for change in changes:
                if change['type'] == 'value_changes':
                    for value_change in change['changes']:
                        column = value_change['column']
                        orig_val = value_change['original_value']
                        mod_val = value_change['modified_value']
                        
                        if column == 'Severity':
                            self._update_severity_patterns(orig_val, mod_val, feedback_record['context'])
                        elif column == 'Occurrence':
                            self._update_occurrence_patterns(orig_val, mod_val, feedback_record['context'])
                        elif column == 'Detection':
                            self._update_detection_patterns(orig_val, mod_val, feedback_record['context'])
            
        except Exception as e:
            self.logger.error(f"Failed to update patterns: {str(e)}")
    
    def _update_severity_patterns(self, original: Any, modified: Any, context: Dict):
        """Update severity adjustment patterns"""
        try:
            # Store severity adjustment patterns
            if 'severity_adjustments' not in self.patterns:
                self.patterns['severity_adjustments'] = {}
            
            pattern_key = f"{original}_to_{modified}"
            if pattern_key not in self.patterns['severity_adjustments']:
                self.patterns['severity_adjustments'][pattern_key] = {
                    'count': 0,
                    'contexts': []
                }
            
            self.patterns['severity_adjustments'][pattern_key]['count'] += 1
            self.patterns['severity_adjustments'][pattern_key]['contexts'].append(context)
            
        except Exception as e:
            self.logger.error(f"Failed to update severity patterns: {str(e)}")
    
    def _update_occurrence_patterns(self, original: Any, modified: Any, context: Dict):
        """Update occurrence adjustment patterns"""
        try:
            if 'occurrence_adjustments' not in self.patterns:
                self.patterns['occurrence_adjustments'] = {}
            
            pattern_key = f"{original}_to_{modified}"
            if pattern_key not in self.patterns['occurrence_adjustments']:
                self.patterns['occurrence_adjustments'][pattern_key] = {
                    'count': 0,
                    'contexts': []
                }
            
            self.patterns['occurrence_adjustments'][pattern_key]['count'] += 1
            self.patterns['occurrence_adjustments'][pattern_key]['contexts'].append(context)
            
        except Exception as e:
            self.logger.error(f"Failed to update occurrence patterns: {str(e)}")
    
    def _update_detection_patterns(self, original: Any, modified: Any, context: Dict):
        """Update detection adjustment patterns"""
        try:
            if 'detection_adjustments' not in self.patterns:
                self.patterns['detection_adjustments'] = {}
            
            pattern_key = f"{original}_to_{modified}"
            if pattern_key not in self.patterns['detection_adjustments']:
                self.patterns['detection_adjustments'][pattern_key] = {
                    'count': 0,
                    'contexts': []
                }
            
            self.patterns['detection_adjustments'][pattern_key]['count'] += 1
            self.patterns['detection_adjustments'][pattern_key]['contexts'].append(context)
            
        except Exception as e:
            self.logger.error(f"Failed to update detection patterns: {str(e)}")
    
    def get_improved_suggestions(self, fmea_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get improved suggestions based on learned patterns"""
        try:
            suggestions = []
            
            if not self.feedback_data:
                return suggestions
            
            # Analyze current FMEA against learned patterns
            for idx, row in fmea_data.iterrows():
                
                # Severity suggestions
                severity_suggestion = self._get_severity_suggestion(row)
                if severity_suggestion:
                    suggestions.append({
                        'type': 'severity_improvement',
                        'row_index': idx,
                        'current_value': row.get('Severity'),
                        'suggested_value': severity_suggestion,
                        'confidence': 0.7,
                        'reason': 'Based on user feedback patterns'
                    })
                
                # Occurrence suggestions
                occurrence_suggestion = self._get_occurrence_suggestion(row)
                if occurrence_suggestion:
                    suggestions.append({
                        'type': 'occurrence_improvement',
                        'row_index': idx,
                        'current_value': row.get('Occurrence'),
                        'suggested_value': occurrence_suggestion,
                        'confidence': 0.7,
                        'reason': 'Based on user feedback patterns'
                    })
                
                # Detection suggestions
                detection_suggestion = self._get_detection_suggestion(row)
                if detection_suggestion:
                    suggestions.append({
                        'type': 'detection_improvement',
                        'row_index': idx,
                        'current_value': row.get('Detection'),
                        'suggested_value': detection_suggestion,
                        'confidence': 0.7,
                        'reason': 'Based on user feedback patterns'
                    })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get improved suggestions: {str(e)}")
            return []
    
    def _get_severity_suggestion(self, row: pd.Series) -> Optional[int]:
        """Get severity suggestion based on patterns"""
        try:
            current_severity = row.get('Severity')
            if current_severity is None:
                return None
            
            # Look for common patterns in severity adjustments
            severity_patterns = self.patterns.get('severity_adjustments', {})
            
            for pattern, data in severity_patterns.items():
                if pattern.startswith(f"{current_severity}_to_") and data['count'] >= 2:
                    suggested_value = int(pattern.split('_to_')[1])
                    if suggested_value != current_severity:
                        return suggested_value
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get severity suggestion: {str(e)}")
            return None
    
    def _get_occurrence_suggestion(self, row: pd.Series) -> Optional[int]:
        """Get occurrence suggestion based on patterns"""
        try:
            current_occurrence = row.get('Occurrence')
            if current_occurrence is None:
                return None
            
            occurrence_patterns = self.patterns.get('occurrence_adjustments', {})
            
            for pattern, data in occurrence_patterns.items():
                if pattern.startswith(f"{current_occurrence}_to_") and data['count'] >= 2:
                    suggested_value = int(pattern.split('_to_')[1])
                    if suggested_value != current_occurrence:
                        return suggested_value
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get occurrence suggestion: {str(e)}")
            return None
    
    def _get_detection_suggestion(self, row: pd.Series) -> Optional[int]:
        """Get detection suggestion based on patterns"""
        try:
            current_detection = row.get('Detection')
            if current_detection is None:
                return None
            
            detection_patterns = self.patterns.get('detection_adjustments', {})
            
            for pattern, data in detection_patterns.items():
                if pattern.startswith(f"{current_detection}_to_") and data['count'] >= 2:
                    suggested_value = int(pattern.split('_to_')[1])
                    if suggested_value != current_detection:
                        return suggested_value
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get detection suggestion: {str(e)}")
            return None
    
    def learn(self, data: Dict) -> Dict[str, Any]:
        """Learn from new data - interface method for coordinator"""
        try:
            return self.store_feedback(
                data.get('user_input', ''),
                data.get('original_fmea'),
                data.get('modified_fmea')
            )
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        try:
            stats = {
                'total_feedback_records': len(self.feedback_data),
                'feedback_types': {},
                'pattern_counts': {},
                'learning_confidence': 0.0
            }
            
            # Count feedback types
            for record in self.feedback_data:
                feedback_type = record.get('feedback_type', 'unknown')
                stats['feedback_types'][feedback_type] = stats['feedback_types'].get(feedback_type, 0) + 1
            
            # Count patterns
            for pattern_type, patterns in self.patterns.items():
                stats['pattern_counts'][pattern_type] = len(patterns)
            
            # Calculate learning confidence
            if len(self.feedback_data) > 0:
                stats['learning_confidence'] = min(1.0, len(self.feedback_data) / 50.0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {str(e)}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': 'LearningAgent',
            'feedback_records': len(self.feedback_data),
            'patterns_learned': sum(len(patterns) for patterns in self.patterns.values()),
            'learning_active': True,
            'active': True
        }


class ReinforcementLearner:
    """
    Advanced reinforcement learning for FMEA optimization
    Implements the ReinforcementLearner from the class diagram
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.state_action_history = []
        
        # Initialize reward system
        self.setup_reward_system()
    
    def setup_reward_system(self):
        """Setup reward system for reinforcement learning"""
        self.reward_system = {
            'rpn_improvement': {
                'high_to_medium': 10,
                'medium_to_low': 5,
                'any_reduction': 2
            },
            'user_acceptance': {
                'accepted': 5,
                'modified': 0,
                'rejected': -3
            },
            'consistency': {
                'consistent_with_history': 3,
                'inconsistent': -1
            }
        }
    
    def get_state(self, fmea_row: pd.Series) -> str:
        """Convert FMEA row to state representation"""
        try:
            # Create state from key features
            component = str(fmea_row.get('Component', 'unknown'))[:10]  # Truncate for state space
            severity = int(fmea_row.get('Severity', 5))
            occurrence = int(fmea_row.get('Occurrence', 5))
            detection = int(fmea_row.get('Detection', 5))
            
            # Categorize RPN
            rpn = severity * occurrence * detection
            if rpn > 200:
                rpn_category = 'critical'
            elif rpn > 100:
                rpn_category = 'high'
            elif rpn > 50:
                rpn_category = 'medium'
            else:
                rpn_category = 'low'
            
            state = f"{component}_{severity}_{occurrence}_{detection}_{rpn_category}"
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to get state: {str(e)}")
            return "unknown_state"
    
    def get_possible_actions(self, current_state: str) -> List[str]:
        """Get possible actions for current state"""
        actions = []
        
        # Possible rating changes (small increments/decrements)
        for rating_type in ['severity', 'occurrence', 'detection']:
            for change in [-2, -1, 1, 2]:
                actions.append(f"adjust_{rating_type}_{change}")
        
        # Special actions
        actions.extend([
            'no_change',
            'add_detection_method',
            'increase_monitoring',
            'recommend_maintenance'
        ])
        
        return actions
    
    def select_action(self, state: str) -> str:
        """Select action using epsilon-greedy strategy"""
        try:
            possible_actions = self.get_possible_actions(state)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.choice(possible_actions)
            else:
                # Exploitation: best known action
                if state in self.q_table:
                    best_action = max(self.q_table[state], key=self.q_table[state].get)
                    return best_action
                else:
                    # If state not seen before, random action
                    return np.random.choice(possible_actions)
                    
        except Exception as e:
            self.logger.error(f"Failed to select action: {str(e)}")
            return 'no_change'
    
    def calculate_reward(self, old_fmea: pd.DataFrame, new_fmea: pd.DataFrame, 
                        user_feedback: str = None) -> float:
        """Calculate reward based on changes"""
        try:
            reward = 0.0
            
            # RPN improvement reward
            if 'RPN' in old_fmea.columns and 'RPN' in new_fmea.columns:
                old_avg_rpn = old_fmea['RPN'].mean()
                new_avg_rpn = new_fmea['RPN'].mean()
                
                rpn_reduction = old_avg_rpn - new_avg_rpn
                if rpn_reduction > 0:
                    # Reward for RPN reduction
                    if old_avg_rpn > 100 and new_avg_rpn <= 100:
                        reward += self.reward_system['rpn_improvement']['high_to_medium']
                    elif old_avg_rpn > 50 and new_avg_rpn <= 50:
                        reward += self.reward_system['rpn_improvement']['medium_to_low']
                    else:
                        reward += self.reward_system['rpn_improvement']['any_reduction']
                elif rpn_reduction < 0:
                    # Penalty for RPN increase
                    reward -= 5
            
            # User feedback reward
            if user_feedback:
                user_feedback_lower = user_feedback.lower()
                if any(word in user_feedback_lower for word in ['good', 'correct', 'accept', 'yes']):
                    reward += self.reward_system['user_acceptance']['accepted']
                elif any(word in user_feedback_lower for word in ['wrong', 'bad', 'reject', 'no']):
                    reward += self.reward_system['user_acceptance']['rejected']
                else:
                    reward += self.reward_system['user_acceptance']['modified']
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Failed to calculate reward: {str(e)}")
            return 0.0
    
    def update_q_table(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-table using Q-learning algorithm"""
        try:
            # Initialize state in Q-table if not exists
            if state not in self.q_table:
                self.q_table[state] = {}
            
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            # Initialize next state
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
                for possible_action in self.get_possible_actions(next_state):
                    self.q_table[next_state][possible_action] = 0.0
            
            # Q-learning update
            old_q_value = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
            
            new_q_value = old_q_value + self.learning_rate * (
                reward + self.discount_factor * max_next_q - old_q_value
            )
            
            self.q_table[state][action] = new_q_value
            
            self.logger.debug(f"Updated Q({state}, {action}): {old_q_value:.3f} -> {new_q_value:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update Q-table: {str(e)}")
    
    def record_changes(self, original_fmea: pd.DataFrame, modified_fmea: pd.DataFrame, 
                      user_feedback: str = None) -> Dict[str, Any]:
        """Record changes for reinforcement learning"""
        try:
            # Calculate reward
            reward = self.calculate_reward(original_fmea, modified_fmea, user_feedback)
            
            # Store experience
            experience = {
                'timestamp': datetime.now().isoformat(),
                'original_size': len(original_fmea),
                'modified_size': len(modified_fmea),
                'reward': reward,
                'user_feedback': user_feedback
            }
            
            self.state_action_history.append(experience)
            
            # Update Q-values for recent state-action pairs
            if len(self.state_action_history) >= 2:
                self._update_recent_experiences(reward)
            
            return {
                'success': True,
                'reward': reward,
                'q_table_size': len(self.q_table)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to record changes: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _update_recent_experiences(self, reward: float):
        """Update Q-values for recent experiences"""
        try:
            # Update last few state-action pairs with the received reward
            # This is a simplified approach - in practice, you'd track specific state-action pairs
            if len(self.state_action_history) >= 2:
                recent_exp = self.state_action_history[-2:]
                
                # Decay epsilon over time to reduce exploration
                self.epsilon = max(0.01, self.epsilon * 0.995)
                
        except Exception as e:
            self.logger.error(f"Failed to update recent experiences: {str(e)}")
    
    def get_recommendations(self, fmea_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI recommendations based on learned Q-values"""
        try:
            recommendations = []
            
            for idx, row in fmea_data.iterrows():
                state = self.get_state(row)
                
                if state in self.q_table:
                    # Get best action for this state
                    best_action = max(self.q_table[state], key=self.q_table[state].get)
                    q_value = self.q_table[state][best_action]
                    
                    # Only recommend if Q-value is positive (learned to be beneficial)
                    if q_value > 0:
                        recommendation = self._action_to_recommendation(best_action, row, q_value)
                        if recommendation:
                            recommendation['row_index'] = idx
                            recommendations.append(recommendation)
            
            # Sort by confidence (Q-value)
            recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {str(e)}")
            return []
    
    def _action_to_recommendation(self, action: str, row: pd.Series, q_value: float) -> Optional[Dict]:
        """Convert action to human-readable recommendation"""
        try:
            if action.startswith('adjust_'):
                parts = action.split('_')
                rating_type = parts[1]
                change = int(parts[2])
                
                current_value = row.get(rating_type.title(), 5)
                new_value = max(1, min(10, current_value + change))
                
                if new_value != current_value:
                    return {
                        'type': 'rating_adjustment',
                        'rating_type': rating_type.title(),
                        'current_value': current_value,
                        'suggested_value': new_value,
                        'confidence': min(1.0, q_value / 10.0),
                        'reason': f'AI learned this adjustment improves outcomes (Q-value: {q_value:.2f})'
                    }
            
            elif action == 'add_detection_method':
                return {
                    'type': 'detection_improvement',
                    'suggestion': 'Consider adding additional detection methods',
                    'confidence': min(1.0, q_value / 10.0),
                    'reason': f'AI learned this improves risk detection (Q-value: {q_value:.2f})'
                }
            
            elif action == 'increase_monitoring':
                return {
                    'type': 'monitoring_improvement',
                    'suggestion': 'Increase monitoring frequency for this component',
                    'confidence': min(1.0, q_value / 10.0),
                    'reason': f'AI learned this reduces risk (Q-value: {q_value:.2f})'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to convert action to recommendation: {str(e)}")
            return None
    
    def save_model(self, filepath: str = "reinforcement_model.pkl"):
        """Save the trained model"""
        try:
            model_data = {
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'state_action_history': self.state_action_history[-100:]  # Keep last 100 for space
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Saved reinforcement learning model to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
    
    def load_model(self, filepath: str = "reinforcement_model.pkl"):
        """Load a trained model"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_table = model_data.get('q_table', {})
                self.learning_rate = model_data.get('learning_rate', 0.1)
                self.discount_factor = model_data.get('discount_factor', 0.9)
                self.epsilon = model_data.get('epsilon', 0.1)
                self.state_action_history = model_data.get('state_action_history', [])
                
                self.logger.info(f"Loaded reinforcement learning model from {filepath}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        try:
            return {
                'q_table_size': len(self.q_table),
                'total_states': len(self.q_table),
                'total_experiences': len(self.state_action_history),
                'current_epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'average_reward': np.mean([exp.get('reward', 0) for exp in self.state_action_history[-50:]]) if self.state_action_history else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning stats: {str(e)}")
            return {}