# agents/ui_controller.py
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

class UIController:
    """
    Controls UI interactions and user interface elements
    Implements the UIController from the class diagram
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ui_state = {}
        self.themes = {
            'light': {
                'primary_color': '#FF6B6B',
                'background_color': '#FFFFFF',
                'secondary_background_color': '#F0F2F6',
                'text_color': '#262730'
            },
            'dark': {
                'primary_color': '#FF6B6B',
                'background_color': '#0E1117',
                'secondary_background_color': '#262730',
                'text_color': '#FAFAFA'
            }
        }
    
    def start_app(self):
        """Initialize and start the application UI"""
        try:
            # Set page configuration
            st.set_page_config(
                page_title="LLM-Integrated FMEA System",
                page_icon="⚠️",
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://github.com/your-repo/fmea-system',
                    'Report a bug': 'https://github.com/your-repo/fmea-system/issues',
                    'About': """
                    # LLM-Integrated FMEA System
                    
                    An AI-powered tool for automated Failure Mode and Effects Analysis.
                    
                    **Features:**
                    - Document processing and analysis
                    - AI-powered FMEA generation
                    - Interactive chat assistant
                    - Risk visualization dashboard
                    - Machine learning optimization
                    
                    Built with ❤️ for reliability engineering.
                    """
                }
            )
            
            # Apply custom CSS
            self._apply_custom_css()
            
            # Initialize UI state
            self._initialize_ui_state()
            
            # Setup sidebar
            self._setup_sidebar()
            
            self.logger.info("UI Controller started successfully")
            
        except Exception as e:
            self.logger.error(f"UI Controller startup failed: {str(e)}")
            st.error(f"Application startup failed: {str(e)}")
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        try:
            custom_css = """
            <style>
            /* Main app styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                padding-top: 1rem;
            }
            
            /* Metrics styling */
            .metric-container {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #ff6b6b;
                margin: 0.5rem 0;
            }
            
            /* Success message styling */
            .success-message {
                background-color: #d4edda;
                color: #155724;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #c3e6cb;
                margin: 1rem 0;
            }
            
            /* Warning message styling */
            .warning-message {
                background-color: #fff3cd;
                color: #856404;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #ffeaa7;
                margin: 1rem 0;
            }
            
            /* Error message styling */
            .error-message {
                background-color: #f8d7da;
                color: #721c24;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #f5c6cb;
                margin: 1rem 0;
            }
            
            /* Chat interface styling */
            .chat-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 1rem;
                border: 1px solid #ddd;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
            }
            
            /* FMEA table styling */
            .fmea-table {
                font-size: 0.9rem;
            }
            
            .fmea-table .rpn-high {
                background-color: #ff6b6b !important;
                color: white !important;
            }
            
            .fmea-table .rpn-medium {
                background-color: #ffa500 !important;
            }
            
            .fmea-table .rpn-low {
                background-color: #28a745 !important;
                color: white !important;
            }
            
            /* Dashboard card styling */
            .dashboard-card {
                background-color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 1rem 0;
            }
            
            /* Risk indicator styling */
            .risk-indicator {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.8rem;
                font-weight: bold;
            }
            
            .risk-critical {
                background-color: #dc3545;
                color: white;
            }
            
            .risk-high {
                background-color: #fd7e14;
                color: white;
            }
            
            .risk-medium {
                background-color: #ffc107;
                color: black;
            }
            
            .risk-low {
                background-color: #28a745;
                color: white;
            }
            
            /* Button styling */
            .stButton > button {
                border-radius: 0.5rem;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            /* File uploader styling */
            .uploadedFile {
                border: 2px dashed #ff6b6b;
                border-radius: 0.5rem;
                padding: 2rem;
                text-align: center;
                background-color: #f8f9fa;
            }
            
            /* Progress bar styling */
            .stProgress > div > div {
                background-color: #ff6b6b;
            }
            
            /* Hide Streamlit footer */
            footer {
                visibility: hidden;
            }
            
            /* Hide hamburger menu */
            #MainMenu {
                visibility: hidden;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .main .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
            </style>
            """
            
            st.markdown(custom_css, unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"CSS application failed: {str(e)}")
    
    def _initialize_ui_state(self):
        """Initialize UI state variables"""
        try:
            if 'ui_initialized' not in st.session_state:
                st.session_state.ui_initialized = True
                st.session_state.current_page = "File Upload"
                st.session_state.sidebar_expanded = True
                st.session_state.theme = "light"
                st.session_state.user_preferences = self._get_default_preferences()
                st.session_state.notification_queue = []
                st.session_state.progress_tasks = {}
                
                # FMEA-specific state
                st.session_state.fmea_data = pd.DataFrame()
                st.session_state.chat_history = []
                st.session_state.uploaded_files = []
                st.session_state.processed_data = {}
                st.session_state.current_fmea = None
                st.session_state.feedback_data = []
                
                # UI state
                st.session_state.filters = {
                    'components': [],
                    'min_rpn': 0,
                    'max_rpn': 1000,
                    'risk_levels': ['Low', 'Medium', 'High', 'Critical']
                }
                
                # Dashboard state
                st.session_state.dashboard_config = {
                    'show_charts': True,
                    'show_tables': True,
                    'chart_type': 'bar',
                    'color_scheme': 'viridis'
                }
                
                self.logger.info("UI state initialized")
                
        except Exception as e:
            self.logger.error(f"UI state initialization failed: {str(e)}")
    
    def _setup_sidebar(self):
        """Setup the application sidebar"""
        try:
            with st.sidebar:
                # App logo and title
                st.markdown("""
                <div style="text-align: center; padding: 1rem 0;">
                    <h2>⚠️ FMEA System</h2>
                    <p style="color: gray; font-size: 0.9rem;">AI-Powered Risk Analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Navigation
                self._render_navigation()
                
                st.divider()
                
                # Quick stats (if FMEA data exists)
                if not st.session_state.fmea_data.empty:
                    self._render_quick_stats()
                    st.divider()
                
                # User preferences
                self._render_user_preferences()
                
                st.divider()
                
                # Help and about
                self._render_help_section()
                
        except Exception as e:
            self.logger.error(f"Sidebar setup failed: {str(e)}")
    
    def _render_navigation(self):
        """Render navigation menu"""
        try:
            st.subheader("📍 Navigation")
            
            pages = {
                "📁 File Upload": "file_upload",
                "🤖 FMEA Generation": "fmea_generation", 
                "💬 Chat Assistant": "chat_assistant",
                "📊 Risk Dashboard": "risk_dashboard",
                "📋 Report Export": "report_export"
            }
            
            # Page selection
            selected_page = st.radio(
                "Select Page:",
                list(pages.keys()),
                index=0 if 'current_page' not in st.session_state else 
                      list(pages.keys()).index(st.session_state.current_page) if st.session_state.current_page in pages.values() else 0,
                key="page_selector"
            )
            
            # Update current page
            st.session_state.current_page = pages[selected_page]
            
            # Page-specific quick actions
            if st.session_state.current_page == "fmea_generation":
                if st.button("🔄 Regenerate FMEA", help="Generate new FMEA from current data"):
                    st.session_state.regenerate_fmea = True
                    
            elif st.session_state.current_page == "chat_assistant":
                if st.button("🗑️ Clear Chat", help="Clear chat history"):
                    st.session_state.chat_history = []
                    st.rerun()
                    
            elif st.session_state.current_page == "risk_dashboard":
                if st.button("📊 Refresh Dashboard", help="Refresh dashboard data"):
                    st.session_state.refresh_dashboard = True
                    
        except Exception as e:
            self.logger.error(f"Navigation rendering failed: {str(e)}")
    
    def _render_quick_stats(self):
        """Render quick statistics in sidebar"""
        try:
            st.subheader("📊 Quick Stats")
            
            fmea_data = st.session_state.fmea_data
            
            if not fmea_data.empty:
                total_entries = len(fmea_data)
                st.metric("Total Entries", total_entries)
                
                if 'RPN' in fmea_data.columns:
                    avg_rpn = fmea_data['RPN'].mean()
                    high_risk_count = len(fmea_data[fmea_data['RPN'] > 100])
                    
                    st.metric("Average RPN", f"{avg_rpn:.1f}")
                    st.metric("High Risk Items", high_risk_count)
                    
                    # Risk distribution mini-chart
                    risk_counts = [
                        len(fmea_data[fmea_data['RPN'] <= 50]),      # Low
                        len(fmea_data[(fmea_data['RPN'] > 50) & (fmea_data['RPN'] <= 100)]),  # Medium
                        len(fmea_data[(fmea_data['RPN'] > 100) & (fmea_data['RPN'] <= 200)]), # High
                        len(fmea_data[fmea_data['RPN'] > 200])       # Critical
                    ]
                    
                    fig = px.pie(
                        values=risk_counts,
                        names=['Low', 'Medium', 'High', 'Critical'],
                        title="Risk Distribution",
                        color_discrete_map={
                            'Low': '#28a745',
                            'Medium': '#ffc107', 
                            'High': '#fd7e14',
                            'Critical': '#dc3545'
                        }
                    )
                    fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            self.logger.error(f"Quick stats rendering failed: {str(e)}")
    
    def _render_user_preferences(self):
        """Render user preferences section"""
        try:
            with st.expander("⚙️ Settings"):
                # Theme preference
                theme = st.selectbox("Theme:", ["Light", "Dark"], index=0)
                st.session_state.theme = theme.lower()
                
                # RPN thresholds
                st.subheader("RPN Thresholds")
                col1, col2 = st.columns(2)
                
                with col1:
                    low_threshold = st.number_input("Low:", value=50, min_value=1, max_value=999)
                    high_threshold = st.number_input("High:", value=100, min_value=low_threshold, max_value=999)
                
                with col2:
                    medium_threshold = st.number_input("Medium:", value=100, min_value=low_threshold, max_value=999)
                    critical_threshold = st.number_input("Critical:", value=200, min_value=high_threshold, max_value=1000)
                
                # Auto-save preferences
                auto_save = st.checkbox("Auto-save FMEA changes", True)
                
                # Notification preferences
                show_hints = st.checkbox("Show helpful hints", True)
                show_notifications = st.checkbox("Show notifications", True)
                
                # Dashboard preferences
                st.subheader("Dashboard")
                default_chart_type = st.selectbox("Default Chart Type:", ["Bar", "Pie", "Scatter", "Heatmap"])
                color_scheme = st.selectbox("Color Scheme:", ["Default", "Viridis", "Plasma", "Reds"])
                
                # Save preferences
                st.session_state.user_preferences = {
                    'theme': theme.lower(),
                    'rpn_thresholds': {
                        'low': low_threshold,
                        'medium': medium_threshold, 
                        'high': high_threshold,
                        'critical': critical_threshold
                    },
                    'auto_save': auto_save,
                    'show_hints': show_hints,
                    'show_notifications': show_notifications,
                    'dashboard': {
                        'default_chart_type': default_chart_type.lower(),
                        'color_scheme': color_scheme.lower()
                    }
                }
                
        except Exception as e:
            self.logger.error(f"User preferences rendering failed: {str(e)}")
    
    def _render_help_section(self):
        """Render help and documentation section"""
        try:
            with st.expander("❓ Help & Support"):
                st.markdown("""
                ### 🚀 Quick Start
                1. **Upload Documents** - Add PDFs, Excel files, or images
                2. **Generate FMEA** - Let AI analyze your documents  
                3. **Chat & Refine** - Use natural language to improve the FMEA
                4. **View Dashboard** - Analyze risks with interactive charts
                5. **Export Report** - Download professional Excel reports
                
                ### 💬 Chat Commands
                - `"Analyze high risk items"`
                - `"Change severity of motor failure to 8"`
                - `"Add bearing failure for spindle"`
                - `"Explain RPN calculation"`
                - `"Suggest improvements"`
                
                ### 📊 RPN Guide
                - **Low Risk**: RPN ≤ 50
                - **Medium Risk**: RPN 51-100
                - **High Risk**: RPN 101-200  
                - **Critical Risk**: RPN > 200
                
                ### 🔧 Supported Files
                - **PDF**: Technical manuals, service bulletins
                - **Excel/CSV**: Historical FMEA data
                - **Images**: Equipment photos (OCR)
                - **Text**: Technical documentation
                """)
                
                if st.button("📖 Open Full Documentation"):
                    st.info("Full documentation available at: docs.fmea-system.com")
                
                if st.button("🐛 Report Issue"):
                    st.info("Report issues at: github.com/fmea-system/issues")
                    
        except Exception as e:
            self.logger.error(f"Help section rendering failed: {str(e)}")
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            'theme': 'light',
            'rpn_thresholds': {'low': 50, 'medium': 100, 'high': 100, 'critical': 200},
            'auto_save': True,
            'show_hints': True,
            'show_notifications': True,
            'dashboard': {
                'default_chart_type': 'bar',
                'color_scheme': 'default'
            }
        }
    
    def handle_upload(self, uploaded_files: List) -> Dict[str, Any]:
        """Handle file upload interactions"""
        try:
            if not uploaded_files:
                return {
                    'success': False,
                    'message': 'No files uploaded'
                }
            
            # Validate file types
            valid_extensions = ['.pdf', '.xlsx', '.csv', '.txt', '.png', '.jpg', '.jpeg', '.xls']
            invalid_files = []
            
            for file in uploaded_files:
                file_extension = '.' + file.name.split('.')[-1].lower()
                if file_extension not in valid_extensions:
                    invalid_files.append(file.name)
            
            if invalid_files:
                return {
                    'success': False,
                    'message': f'Invalid file types: {", ".join(invalid_files)}',
                    'invalid_files': invalid_files
                }
            
            # Validate file sizes (max 10MB per file)
            large_files = []
            for file in uploaded_files:
                if file.size > 10 * 1024 * 1024:  # 10MB
                    large_files.append(file.name)
            
            if large_files:
                return {
                    'success': False,
                    'message': f'Files too large (>10MB): {", ".join(large_files)}',
                    'large_files': large_files
                }
            
            # Store files in session state
            st.session_state.uploaded_files = uploaded_files
            
            return {
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} files',
                'file_count': len(uploaded_files),
                'files': [{'name': f.name, 'size': f.size, 'type': f.type} for f in uploaded_files]
            }
            
        except Exception as e:
            self.logger.error(f"File upload handling failed: {str(e)}")
            return {
                'success': False,
                'message': f'Upload failed: {str(e)}'
            }
    
    def show_chat(self, chat_history: List[Dict], current_fmea: pd.DataFrame):
        """Display chat interface"""
        try:
            st.subheader("💬 Chat with FMEA Assistant")
            
            # Chat container with custom styling
            chat_container = st.container()
            
            with chat_container:
                # Display existing chat history
                for message in chat_history:
                    with st.chat_message(message.get("role", "user")):
                        content = message.get("content", "")
                        if content:
                            st.write(content)
                        
                        # Show any attachments or data
                        if "data" in message:
                            with st.expander("📊 View Data"):
                                st.json(message["data"])
                
                # Chat input area
                st.markdown("---")
                
                # Quick action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔍 Analyze Risks", help="Analyze current high-risk items"):
                        return {"action": "analyze_risks"}
                
                with col2:
                    if st.button("💡 Get Suggestions", help="Get improvement suggestions"):
                        return {"action": "get_suggestions"}
                
                with col3:
                    if st.button("✅ Validate FMEA", help="Check FMEA completeness"):
                        return {"action": "validate_fmea"}
                
                with col4:
                    if st.button("❓ Explain RPN", help="Explain RPN calculation"):
                        return {"action": "explain_rpn"}
                
                # Chat input (handled by main app)
                return {"action": "display_only"}
                
        except Exception as e:
            self.logger.error(f"Chat display failed: {str(e)}")
            st.error(f"Chat interface error: {str(e)}")
            return {"action": "error", "error": str(e)}
    
    def show_dashboard(self, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Display dashboard interface with filters and controls"""
        try:
            if fmea_data.empty:
                st.warning("⚠️ No FMEA data available for dashboard display")
                return {'filtered_data': fmea_data, 'show_charts': True, 'show_tables': True}
            
            # Dashboard header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader("📊 Risk Analytics Dashboard")
            
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            with col3:
                if st.button("⚙️ Configure"):
                    st.session_state.show_dashboard_config = not st.session_state.get('show_dashboard_config', False)
            
            # Dashboard configuration panel
            if st.session_state.get('show_dashboard_config', False):
                self._show_dashboard_config()
            
            # Filters section
            with st.sidebar:
                st.subheader("🎛️ Dashboard Filters")
                
                # Component filter
                if 'Component' in fmea_data.columns:
                    components = ['All'] + sorted(list(fmea_data['Component'].unique()))
                    selected_components = st.multiselect(
                        "Components:",
                        components,
                        default=['All'],
                        help="Filter by specific components"
                    )
                else:
                    selected_components = ['All']
                
                # RPN range filter
                if 'RPN' in fmea_data.columns:
                    min_rpn_data = int(fmea_data['RPN'].min())
                    max_rpn_data = int(fmea_data['RPN'].max())
                    
                    rpn_range = st.slider(
                        "RPN Range:",
                        min_value=min_rpn_data,
                        max_value=max_rpn_data,
                        value=(min_rpn_data, max_rpn_data),
                        help="Filter by RPN value range"
                    )
                else:
                    rpn_range = (0, 1000)
                
                # Risk level filter
                risk_levels = st.multiselect(
                    "Risk Levels:",
                    ['Low', 'Medium', 'High', 'Critical'],
                    default=['Low', 'Medium', 'High', 'Critical'],
                    help="Filter by risk classification"
                )
                
                # Date range filter (if available)
                if 'Date' in fmea_data.columns:
                    date_filter = st.date_input(
                        "Date Range:",
                        help="Filter by date range"
                    )
                
                # Display options
                st.subheader("📋 Display Options")
                show_charts = st.checkbox("Show Charts", True)
                show_tables = st.checkbox("Show Tables", True)
                show_summary = st.checkbox("Show Summary", True)
                
                # Chart options
                if show_charts:
                    chart_type = st.selectbox(
                        "Chart Type:",
                        ["Bar", "Pie", "Scatter", "Heatmap", "Treemap"],
                        help="Select chart visualization type"
                    )
                    
                    color_scheme = st.selectbox(
                        "Color Scheme:",
                        ["Default", "Viridis", "Plasma", "Reds", "Blues"],
                        help="Select color scheme for charts"
                    )
            
            # Apply filters
            filtered_data = self._apply_dashboard_filters(
                fmea_data, selected_components, rpn_range, risk_levels
            )
            
            # Show filter results
            if len(filtered_data) != len(fmea_data):
                st.info(f"📊 Showing {len(filtered_data)} of {len(fmea_data)} entries based on filters")
            
            return {
                'filtered_data': filtered_data,
                'show_charts': show_charts,
                'show_tables': show_tables,
                'show_summary': show_summary,
                'chart_type': chart_type if show_charts else None,
                'color_scheme': color_scheme if show_charts else None
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard display failed: {str(e)}")
            st.error(f"Dashboard error: {str(e)}")
            return {'filtered_data': fmea_data, 'show_charts': True, 'show_tables': True}
    
    def _show_dashboard_config(self):
        """Show dashboard configuration panel"""
        try:
            with st.expander("⚙️ Dashboard Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Chart Settings")
                    auto_refresh = st.checkbox("Auto-refresh", False)
                    animation_enabled = st.checkbox("Enable animations", True)
                    show_grid = st.checkbox("Show grid lines", True)
                    
                with col2:
                    st.subheader("Table Settings")
                    page_size = st.number_input("Rows per page:", min_value=5, max_value=100, value=20)
                    sortable = st.checkbox("Sortable columns", True)
                    filterable = st.checkbox("Filterable columns", True)
                
                # Export settings
                st.subheader("Export Settings")
                include_metadata = st.checkbox("Include metadata in exports", True)
                high_res_charts = st.checkbox("High-resolution charts", False)
                
                if st.button("💾 Save Configuration"):
                    st.session_state.dashboard_config.update({
                        'auto_refresh': auto_refresh,
                        'animation_enabled': animation_enabled,
                        'show_grid': show_grid,
                        'page_size': page_size,
                        'sortable': sortable,
                        'filterable': filterable,
                        'include_metadata': include_metadata,
                        'high_res_charts': high_res_charts
                    })
                    st.success("✅ Configuration saved!")
                    
        except Exception as e:
            self.logger.error(f"Dashboard config display failed: {str(e)}")
    
    def _apply_dashboard_filters(self, fmea_data: pd.DataFrame, components: List[str], 
                                rpn_range: Tuple[int, int], risk_levels: List[str]) -> pd.DataFrame:
        """Apply dashboard filters to FMEA data"""
        try:
            filtered_data = fmea_data.copy()
            
            # Apply component filter
            if 'All' not in components and 'Component' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Component'].isin(components)]
            
            # Apply RPN range filter
            if 'RPN' in filtered_data.columns:
                filtered_data = filtered_data[
                    (filtered_data['RPN'] >= rpn_range[0]) & 
                    (filtered_data['RPN'] <= rpn_range[1])
                ]
            
            # Apply risk level filter
            if 'RPN' in filtered_data.columns and risk_levels:
                risk_mask = pd.Series([False] * len(filtered_data))
                
                for risk_level in risk_levels:
                    if risk_level == 'Low':
                        risk_mask |= filtered_data['RPN'] <= 50
                    elif risk_level == 'Medium':
                        risk_mask |= (filtered_data['RPN'] > 50) & (filtered_data['RPN'] <= 100)
                    elif risk_level == 'High':
                        risk_mask |= (filtered_data['RPN'] > 100) & (filtered_data['RPN'] <= 200)
                    elif risk_level == 'Critical':
                        risk_mask |= filtered_data['RPN'] > 200
                
                filtered_data = filtered_data[risk_mask]
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Filter application failed: {str(e)}")
            return fmea_data
    
    def accept_user_input(self, input_type: str = "text", **kwargs) -> Optional[str]:
        """Accept various types of user input"""
        try:
            if input_type == "text":
                placeholder = kwargs.get('placeholder', 'Enter your input...')
                help_text = kwargs.get('help', None)
                return st.text_input("Input:", placeholder=placeholder, help=help_text)
                
            elif input_type == "textarea":
                placeholder = kwargs.get('placeholder', 'Enter your input...')
                height = kwargs.get('height', 100)
                return st.text_area("Input:", placeholder=placeholder, height=height)
                
            elif input_type == "chat":
                placeholder = kwargs.get('placeholder', 'Type your message...')
                return st.chat_input(placeholder)
                
            elif input_type == "number":
                min_val = kwargs.get('min_value', 0)
                max_val = kwargs.get('max_value', 100)
                default = kwargs.get('value', min_val)
                return st.number_input("Input:", min_value=min_val, max_value=max_val, value=default)
                
            elif input_type == "slider":
                min_val = kwargs.get('min_value', 0)
                max_val = kwargs.get('max_value', 100)
                default = kwargs.get('value', min_val)
                return st.slider("Input:", min_value=min_val, max_value=max_val, value=default)
                
            elif input_type == "selectbox":
                options = kwargs.get('options', ['Option 1', 'Option 2'])
                index = kwargs.get('index', 0)
                return st.selectbox("Select:", options, index=index)
                
            elif input_type == "multiselect":
                options = kwargs.get('options', ['Option 1', 'Option 2'])
                default = kwargs.get('default', [])
                return st.multiselect("Select multiple:", options, default=default)
                
            elif input_type == "file":
                file_types = kwargs.get('file_types', None)
                accept_multiple = kwargs.get('accept_multiple_files', False)
                return st.file_uploader("Upload file:", type=file_types, accept_multiple_files=accept_multiple)
                
            else:
                self.logger.warning(f"Unknown input type: {input_type}")
                return st.text_input("Enter your input:")
                
        except Exception as e:
            self.logger.error(f"User input acceptance failed: {str(e)}")
            return None
    
    def display_fmea_table(self, fmea_data: pd.DataFrame, editable: bool = True) -> pd.DataFrame:
        """Display FMEA table with optional editing capabilities"""
        try:
            if fmea_data.empty:
                st.info("📝 No FMEA data to display")
                return fmea_data
            
            # Add custom CSS for table styling
            st.markdown("""
            <style>
            .fmea-table {
                font-size: 0.9rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if editable:
                st.subheader("📋 FMEA Table (Editable)")
                
                # Editable data editor with enhanced configuration
                edited_df = st.data_editor(
                    fmea_data,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "Component": st.column_config.TextColumn(
                            "Component/System",
                            help="Component or system being analyzed",
                            max_chars=50
                        ),
                        "Failure Mode": st.column_config.TextColumn(
                            "Failure Mode",
                            help="How the component can fail",
                            max_chars=100
                        ),
                        "Effect": st.column_config.TextColumn(
                            "Effect of Failure",
                            help="Impact when failure occurs",
                            max_chars=150
                        ),
                        "Cause": st.column_config.TextColumn(
                            "Potential Cause",
                            help="What causes the failure",
                            max_chars=150
                        ),
                        "Detection Method": st.column_config.TextColumn(
                            "Detection Method",
                            help="How failure is detected",
                            max_chars=100
                        ),
                        "Severity": st.column_config.NumberColumn(
                            "Severity (1-10)",
                            min_value=1,
                            max_value=10,
                            step=1,
                            format="%d",
                            help="Impact severity rating (1=minimal, 10=catastrophic)"
                        ),
                        "Occurrence": st.column_config.NumberColumn(
                            "Occurrence (1-10)",
                            min_value=1,
                            max_value=10,
                            step=1,
                            format="%d",
                            help="Likelihood of occurrence (1=remote, 10=almost certain)"
                        ),
                        "Detection": st.column_config.NumberColumn(
                            "Detection (1-10)",
                            min_value=1,
                            max_value=10,
                            step=1,
                            format="%d",
                            help="Detection probability (1=almost certain, 10=almost impossible)"
                        ),
                        "RPN": st.column_config.NumberColumn(
                            "RPN",
                            disabled=True,
                            help="Risk Priority Number (auto-calculated: S × O × D)"
                        ),
                        "Source": st.column_config.SelectboxColumn(
                            "Source",
                            options=["Document Extraction", "User Input", "Knowledge Base", "Historical Data"],
                            help="Source of this FMEA entry"
                        )
                    },
                    key="fmea_editor"
                )
                
                # Auto-calculate RPN if SOD columns exist and values changed
                if all(col in edited_df.columns for col in ['Severity', 'Occurrence', 'Detection']):
                    edited_df['RPN'] = edited_df['Severity'] * edited_df['Occurrence'] * edited_df['Detection']
                
                # Show table statistics
                self._show_table_statistics(edited_df)
                
                # Table actions
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("📊 Sort by RPN", help="Sort table by RPN values"):
                        edited_df = edited_df.sort_values('RPN', ascending=False)
                        st.rerun()
                
                with col2:
                    if st.button("🎯 Highlight Risks", help="Highlight high-risk items"):
                        st.session_state.highlight_risks = True
                
                with col3:
                    if st.button("📋 Add Entry", help="Add new FMEA entry"):
                        st.session_state.show_add_entry_form = True
                
                with col4:
                    if st.button("🗑️ Clear Table", help="Clear all entries"):
                        if st.session_state.get('confirm_clear', False):
                            edited_df = pd.DataFrame(columns=fmea_data.columns)
                            st.session_state.confirm_clear = False
                            st.rerun()
                        else:
                            st.session_state.confirm_clear = True
                            st.warning("⚠️ Click again to confirm deletion")
                
                # Show add entry form if requested
                if st.session_state.get('show_add_entry_form', False):
                    self._show_add_entry_form(edited_df)
                
                return edited_df
                
            else:
                # Read-only display with enhanced formatting
                st.subheader("📋 FMEA Table (Read-Only)")
                
                # Apply conditional formatting
                styled_df = self._apply_table_styling(fmea_data)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )
                
                return fmea_data
                
        except Exception as e:
            self.logger.error(f"FMEA table display failed: {str(e)}")
            st.error(f"Table display error: {str(e)}")
            return fmea_data
    
    def _show_table_statistics(self, fmea_data: pd.DataFrame):
        """Show table statistics below the FMEA table"""
        try:
            if not fmea_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Entries", len(fmea_data))
                
                if 'RPN' in fmea_data.columns:
                    with col2:
                        avg_rpn = fmea_data['RPN'].mean()
                        st.metric("Average RPN", f"{avg_rpn:.1f}")
                    
                    with col3:
                        high_risk_count = len(fmea_data[fmea_data['RPN'] > 100])
                        st.metric("High Risk Items", high_risk_count)
                    
                    with col4:
                        max_rpn = fmea_data['RPN'].max()
                        st.metric("Maximum RPN", int(max_rpn))
                        
        except Exception as e:
            self.logger.error(f"Table statistics display failed: {str(e)}")
    
    def _apply_table_styling(self, fmea_data: pd.DataFrame) -> pd.DataFrame:
        """Apply conditional formatting to FMEA table"""
        try:
            if 'RPN' not in fmea_data.columns:
                return fmea_data
            
            def highlight_rpn(val):
                """Color-code RPN values"""
                if pd.isna(val):
                    return ''
                elif val > 200:
                    return 'background-color: #ff6b6b; color: white'  # Critical - Red
                elif val > 100:
                    return 'background-color: #ffa500; color: white'  # High - Orange
                elif val > 50:
                    return 'background-color: #ffff00; color: black'  # Medium - Yellow
                else:
                    return 'background-color: #90EE90; color: black'  # Low - Light Green
            
            # Apply styling
            styled_df = fmea_data.style.applymap(highlight_rpn, subset=['RPN'])
            
            return styled_df
            
        except Exception as e:
            self.logger.error(f"Table styling failed: {str(e)}")
            return fmea_data
    
    def _show_add_entry_form(self, current_fmea: pd.DataFrame):
        """Show form to add new FMEA entry"""
        try:
            with st.form("add_fmea_entry"):
                st.subheader("➕ Add New FMEA Entry")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    component = st.text_input("Component/System*", help="Required field")
                    failure_mode = st.text_input("Failure Mode*", help="Required field")
                    effect = st.text_area("Effect of Failure", height=80)
                    cause = st.text_area("Potential Cause", height=80)
                
                with col2:
                    detection_method = st.text_area("Detection Method", height=80)
                    
                    severity = st.slider("Severity (1-10)", 1, 10, 5, 
                                       help="1=No impact, 10=Catastrophic")
                    occurrence = st.slider("Occurrence (1-10)", 1, 10, 5,
                                         help="1=Remote, 10=Almost certain")
                    detection = st.slider("Detection (1-10)", 1, 10, 5,
                                        help="1=Almost certain detection, 10=Cannot detect")
                
                # Calculate RPN
                rpn = severity * occurrence * detection
                st.info(f"Calculated RPN: {rpn}")
                
                # Form submission
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.form_submit_button("✅ Add Entry", type="primary"):
                        if component and failure_mode:
                            new_entry = {
                                'Component': component,
                                'Failure Mode': failure_mode,
                                'Effect': effect or 'To be determined',
                                'Cause': cause or 'To be determined',
                                'Detection Method': detection_method or 'To be determined',
                                'Severity': severity,
                                'Occurrence': occurrence,
                                'Detection': detection,
                                'RPN': rpn,
                                'Source': 'User Input'
                            }
                            
                            # Add to current FMEA
                            new_df = pd.concat([current_fmea, pd.DataFrame([new_entry])], ignore_index=True)
                            st.session_state.fmea_data = new_df
                            st.session_state.show_add_entry_form = False
                            st.success("✅ Entry added successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Component and Failure Mode are required fields")
                
                with col2:
                    if st.form_submit_button("🔄 Reset Form"):
                        st.rerun()
                
                with col3:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.show_add_entry_form = False
                        st.rerun()
                        
        except Exception as e:
            self.logger.error(f"Add entry form display failed: {str(e)}")
    
    def show_export_options(self, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Display export options and controls"""
        try:
            st.subheader("📤 Export FMEA Report")
            
            if fmea_data.empty:
                st.warning("⚠️ No FMEA data available for export")
                return {}
            
            export_options = {}
            
            # Export format selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Export Format")
                export_format = st.selectbox(
                    "Choose format:",
                    ["Excel (.xlsx)", "CSV (.csv)", "PDF Report", "JSON Data"],
                    help="Select the export format for your FMEA report"
                )
                export_options['format'] = export_format
                
                # File naming
                default_name = f"FMEA_Report_{datetime.now().strftime('%Y%m%d_%H%M')}"
                filename = st.text_input("Filename (without extension):", default_name)
                export_options['filename'] = filename
            
            with col2:
                st.markdown("### ⚙️ Export Options")
                include_charts = st.checkbox("Include Charts", True, help="Add visual charts to the report")
                include_summary = st.checkbox("Include Summary", True, help="Add executive summary")
                include_raw_data = st.checkbox("Include Raw Data", True, help="Include complete FMEA table")
                include_metadata = st.checkbox("Include Metadata", False, help="Add creation date, user info, etc.")
                
                export_options.update({
                    'include_charts': include_charts,
                    'include_summary': include_summary,
                    'include_raw_data': include_raw_data,
                    'include_metadata': include_metadata
                })
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    filter_high_risk = st.checkbox("Export only high-risk items (RPN > 100)", False)
                    sort_by_rpn = st.checkbox("Sort by RPN (descending)", True)
                    include_calculations = st.checkbox("Show RPN calculations", False)
                
                with col2:
                    compress_file = st.checkbox("Compress output file", False)
                    password_protect = st.checkbox("Password protect (Excel only)", False)
                    if password_protect and export_format == "Excel (.xlsx)":
                        password = st.text_input("Password:", type="password")
                        export_options['password'] = password
                
                export_options.update({
                    'filter_high_risk': filter_high_risk,
                    'sort_by_rpn': sort_by_rpn,
                    'include_calculations': include_calculations,
                    'compress_file': compress_file,
                    'password_protect': password_protect
                })
            
            # Preview section
            st.markdown("### 👀 Export Preview")
            
            # Show what will be exported
            preview_data = fmea_data.copy()
            
            if export_options.get('filter_high_risk', False) and 'RPN' in preview_data.columns:
                preview_data = preview_data[preview_data['RPN'] > 100]
            
            if export_options.get('sort_by_rpn', False) and 'RPN' in preview_data.columns:
                preview_data = preview_data.sort_values('RPN', ascending=False)
            
            # Show preview statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entries to Export", len(preview_data))
            
            if 'RPN' in preview_data.columns and not preview_data.empty:
                with col2:
                    st.metric("Average RPN", f"{preview_data['RPN'].mean():.1f}")
                
                with col3:
                    st.metric("Max RPN", int(preview_data['RPN'].max()))
                
                with col4:
                    high_risk_count = len(preview_data[preview_data['RPN'] > 100])
                    st.metric("High Risk Items", high_risk_count)
            
            # Show preview table (first 5 rows)
            if not preview_data.empty:
                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(preview_data.head(), use_container_width=True)
            
            return export_options
            
        except Exception as e:
            self.logger.error(f"Export options display failed: {str(e)}")
            return {'format': 'Excel (.xlsx)', 'filename': 'FMEA_Report'}
    
    def show_progress(self, message: str, progress: float = None):
        """Show progress indicator"""
        try:
            progress_container = st.container()
            
            with progress_container:
                if progress is not None:
                    # Show progress bar with percentage
                    progress_bar = st.progress(progress)
                    st.write(f"{message} ({progress*100:.0f}%)")
                else:
                    # Show spinner
                    with st.spinner(message):
                        pass
            
        except Exception as e:
            self.logger.error(f"Progress display failed: {str(e)}")
    
    def show_notification(self, message: str, notification_type: str = "info", duration: int = 5):
        """Show notification to user"""
        try:
            # Add to notification queue
            notification = {
                'message': message,
                'type': notification_type,
                'timestamp': datetime.now(),
                'duration': duration
            }
            
            if 'notification_queue' not in st.session_state:
                st.session_state.notification_queue = []
            
            st.session_state.notification_queue.append(notification)
            
            # Display notification
            if notification_type == "success":
                st.success(f"✅ {message}")
            elif notification_type == "warning":
                st.warning(f"⚠️ {message}")
            elif notification_type == "error":
                st.error(f"❌ {message}")
            elif notification_type == "info":
                st.info(f"ℹ️ {message}")
            else:
                st.write(f"🔔 {message}")
                
        except Exception as e:
            self.logger.error(f"Notification display failed: {str(e)}")
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences and settings"""
        try:
            return st.session_state.get('user_preferences', self._get_default_preferences())
        except Exception as e:
            self.logger.error(f"User preferences retrieval failed: {str(e)}")
            return self._get_default_preferences()
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        try:
            st.session_state.user_preferences.update(preferences)
            self.show_notification("Settings saved successfully!", "success")
        except Exception as e:
            self.logger.error(f"User preferences update failed: {str(e)}")
            self.show_notification("Failed to save settings", "error")
    
    def render_help_section(self):
        """Render comprehensive help and documentation section"""
        try:
            st.markdown("### 📚 Help & Documentation")
            
            # Create tabs for different help sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Quick Start", "💬 Chat Guide", "📊 Dashboard", "🔧 Troubleshooting", "❓ FAQ"])
            
            with tab1:
                st.markdown("""
                #### 🚀 Getting Started with FMEA System
                
                **Step 1: Upload Documents**
                - Drag and drop or click to upload technical documents
                - Supported: PDF manuals, Excel/CSV files, images, text files
                - Max file size: 10MB per file
                
                **Step 2: Generate FMEA** 
                - Click "Generate FMEA" to start AI analysis
                - System will identify components, failure modes, and risks
                - Initial RPN values are estimated and should be reviewed
                
                **Step 3: Review and Refine**
                - Use the interactive table to edit values
                - Adjust Severity, Occurrence, and Detection ratings
                - RPN is automatically calculated (S × O × D)
                
                **Step 4: Chat Assistance**
                - Ask natural language questions about your FMEA
                - Request specific changes: "Change motor severity to 8"
                - Get explanations: "What does RPN mean?"
                
                **Step 5: Analyze Risks**
                - View the dashboard for visual risk analysis
                - Identify high-priority items requiring action
                - Filter and sort data by different criteria
                
                **Step 6: Export Report**
                - Generate professional Excel reports with charts
                - Include executive summary and recommendations
                - Save as CSV for further analysis
                """)
            
            with tab2:
                st.markdown("""
                #### 💬 Chat Assistant Guide
                
                **Analysis Commands:**
                - `"Analyze high risk items"` - Get analysis of RPN > 100 items
                - `"Show component breakdown"` - View failure modes by component
                - `"What are the top risks?"` - List highest RPN items
                
                **Modification Commands:**
                - `"Change severity of [item] to [number]"` - Adjust severity rating
                - `"Set occurrence for motor failure to 6"` - Update occurrence
                - `"Update detection for bearing wear to 4"` - Modify detection
                
                **Addition Commands:**
                - `"Add bearing failure for spindle motor"` - Add new entry
                - `"Create entry for pump malfunction"` - Add component failure
                
                **Information Commands:**
                - `"Explain RPN calculation"` - Learn about Risk Priority Numbers
                - `"What does severity mean?"` - Understand rating scales
                - `"How do I reduce risk?"` - Get improvement suggestions
                
                **Validation Commands:**
                - `"Check FMEA completeness"` - Validate all required fields
                - `"Find missing information"` - Identify incomplete entries
                - `"Suggest improvements"` - Get AI recommendations
                """)
            
            with tab3:
                st.markdown("""
                #### 📊 Dashboard Guide
                
                **Risk Metrics:**
                - **Total Entries**: Number of failure modes analyzed
                - **Average RPN**: Mean risk priority across all items
                - **High Risk Items**: Count of items with RPN > 100
                - **Maximum RPN**: Highest individual risk score
                
                **Chart Types:**
                - **Bar Charts**: Compare RPN values across components
                - **Pie Charts**: Show risk distribution by category
                - **Scatter Plots**: Visualize Severity vs Occurrence
                - **Heatmaps**: Display risk intensity patterns
                
                **Filters Available:**
                - **Component Filter**: Show specific components only
                - **RPN Range**: Filter by risk priority range
                - **Risk Level**: Filter by Low/Medium/High/Critical
                - **Date Range**: Filter by creation/modification date
                
                **Interactive Features:**
                - Click chart elements to drill down
                - Hover for detailed information
                - Export charts as images
                - Customize colors and themes
                """)
            
            with tab4:
                st.markdown("""
                #### 🔧 Troubleshooting
                
                **File Upload Issues:**
                - ✅ Check file size (max 10MB)
                - ✅ Verify file format (PDF, Excel, CSV, images)
                - ✅ Ensure files are not corrupted or password-protected
                - ✅ Try uploading one file at a time
                
                **FMEA Generation Problems:**
                - ✅ Ensure documents contain technical content
                - ✅ Check that text is readable (not scanned images)
                - ✅ Verify document language is English
                - ✅ Try with smaller files first
                
                **Chat Not Responding:**
                - ✅ Check that FMEA data exists
                - ✅ Try rephrasing your question
                - ✅ Use specific component/failure mode names
                - ✅ Refresh the page and try again
                
                **Dashboard Not Loading:**
                - ✅ Ensure FMEA data is available
                - ✅ Check browser compatibility (Chrome, Firefox, Safari)
                - ✅ Clear browser cache and reload
                - ✅ Disable browser ad blockers
                
                **Export Failures:**
                - ✅ Verify FMEA data exists
                - ✅ Check available disk space
                - ✅ Try different export format
                - ✅ Disable password protection if enabled
                """)
            
            with tab5:
                st.markdown("""
                #### ❓ Frequently Asked Questions
                
                **Q: What file types are supported?**
                A: PDF documents, Excel files (.xlsx, .xls), CSV files, text files (.txt), and images (.png, .jpg, .jpeg).
                
                **Q: How accurate are the AI-generated FMEA entries?**
                A: AI provides good starting points, but all entries should be reviewed and validated by subject matter experts.
                
                **Q: Can I import existing FMEA data?**
                A: Yes, upload Excel or CSV files with existing FMEA data. The system will recognize standard FMEA columns.
                
                **Q: What do the RPN numbers mean?**
                A: RPN (Risk Priority Number) = Severity × Occurrence × Detection. Higher numbers indicate higher priority risks.
                
                **Q: How do I backup my FMEA data?**
                A: Use the Export function to save your FMEA as Excel or CSV files. These can be re-imported later.
                
                **Q: Can multiple people work on the same FMEA?**
                A: Currently, this is a single-user application. Export/import files to share with team members.
                
                **Q: How does the learning system work?**
                A: The system learns from your corrections and preferences to improve future suggestions and automate repetitive tasks.
                
                **Q: Is my data secure?**
                A: Yes, all processing happens locally on your computer. No data is sent to external servers.
                """)
                
        except Exception as e:
            self.logger.error(f"Help section rendering failed: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get UI controller status"""
        try:
            return {
                'name': 'UIController',
                'ui_initialized': st.session_state.get('ui_initialized', False),
                'current_page': st.session_state.get('current_page', 'Unknown'),
                'theme': st.session_state.get('theme', 'light'),
                'notifications_pending': len(st.session_state.get('notification_queue', [])),
                'fmea_entries': len(st.session_state.get('fmea_data', pd.DataFrame())),
                'active': True
            }
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {str(e)}")
            return {
                'name': 'UIController', 
                'active': False,
                'error': str(e)
            }