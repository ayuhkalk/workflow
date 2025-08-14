# main.py - Main Application Entry Point
import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Import custom modules
from agents.ui_controller import UIController
from agents.agent_coordinator import AgentCoordinator
from agents.document_agent import DocumentAgent
from agents.fmea_agent import FMEAAgent
from agents.chat_agent import ChatAgent
from agents.visual_dashboard import VisualDashboard
from agents.report_generator_agent import ReportGeneratorAgent
from agents.learning_agent import LearningAgent

class FMEASystem:
    """Main FMEA System class that orchestrates all components"""
    
    def __init__(self):
        self.initialize_session_state()
        self.coordinator = AgentCoordinator()
        self.ui_controller = UIController()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'fmea_data' not in st.session_state:
            st.session_state.fmea_data = pd.DataFrame()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'current_fmea' not in st.session_state:
            st.session_state.current_fmea = None
        if 'feedback_data' not in st.session_state:
            st.session_state.feedback_data = []
            
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="LLM-Integrated FMEA System",
            page_icon="⚠️",
            layout="wide"
        )
        
        st.title("🤖 LLM-Integrated FMEA System")
        st.markdown("Automated Failure Mode and Effects Analysis with AI Assistance")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select Page:",
                ["📁 File Upload", "🤖 FMEA Generation", "💬 Chat Assistant", 
                 "📊 Risk Dashboard", "📋 Report Export"]
            )
        
        # Route to appropriate page
        if page == "📁 File Upload":
            self.file_upload_page()
        elif page == "🤖 FMEA Generation":
            self.fmea_generation_page()
        elif page == "💬 Chat Assistant":
            self.chat_assistant_page()
        elif page == "📊 Risk Dashboard":
            self.risk_dashboard_page()
        elif page == "📋 Report Export":
            self.report_export_page()
    
    def file_upload_page(self):
        """File upload and processing page"""
        st.header("📁 Document Upload & Processing")
        
        uploaded_files = st.file_uploader(
            "Upload technical documents, manuals, or historical FMEA files",
            type=['pdf', 'xlsx', 'csv', 'txt', 'png', 'jpg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        processed_data = self.coordinator.process_documents(uploaded_files)
                        st.session_state.processed_data = processed_data
                        st.success("Documents processed successfully!")
                        
                        # Display processed data summary
                        self.display_processed_data_summary(processed_data)
            
            with col2:
                if st.button("🗑️ Clear Files"):
                    st.session_state.uploaded_files = []
                    st.rerun()
        
        # Display current files
        if st.session_state.uploaded_files:
            st.subheader("📋 Uploaded Files:")
            for file in st.session_state.uploaded_files:
                st.write(f"• {file.name} ({file.type})")
    
    def display_processed_data_summary(self, processed_data):
        """Display summary of processed data"""
        st.subheader("📊 Processing Results")
        
        if 'extracted_text' in processed_data:
            st.write(f"**Text extracted:** {len(processed_data['extracted_text'])} characters")
        
        if 'tables' in processed_data:
            st.write(f"**Tables found:** {len(processed_data['tables'])}")
            
        if 'failure_modes' in processed_data:
            st.write(f"**Potential failure modes identified:** {len(processed_data['failure_modes'])}")
            
        # Show preview of extracted data
        with st.expander("🔍 View Extracted Data Preview"):
            if 'extracted_text' in processed_data:
                st.text_area("Extracted Text (first 500 chars):", 
                           processed_data['extracted_text'][:500], height=150)
    
    def fmea_generation_page(self):
        """FMEA generation and editing page"""
        st.header("🤖 FMEA Generation")
        
        if not hasattr(st.session_state, 'processed_data'):
            st.warning("⚠️ Please upload and process documents first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Generate FMEA", type="primary"):
                with st.spinner("Generating FMEA..."):
                    fmea_data = self.coordinator.generate_fmea(st.session_state.processed_data)
                    st.session_state.fmea_data = fmea_data
                    st.session_state.current_fmea = fmea_data
                    st.success("FMEA generated successfully!")
        
        with col2:
            if st.button("➕ Add Manual Entry"):
                self.show_manual_entry_form()
        
        # Display FMEA table if available
        if not st.session_state.fmea_data.empty:
            st.subheader("📋 Generated FMEA Table")
            
            # Make the dataframe editable
            edited_df = st.data_editor(
                st.session_state.fmea_data,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Severity": st.column_config.NumberColumn(
                        "Severity (1-10)",
                        min_value=1,
                        max_value=10,
                        step=1
                    ),
                    "Occurrence": st.column_config.NumberColumn(
                        "Occurrence (1-10)",
                        min_value=1,
                        max_value=10,
                        step=1
                    ),
                    "Detection": st.column_config.NumberColumn(
                        "Detection (1-10)",
                        min_value=1,
                        max_value=10,
                        step=1
                    )
                }
            )
            
            # Update session state if changes made
            if not edited_df.equals(st.session_state.fmea_data):
                st.session_state.fmea_data = edited_df
                st.session_state.current_fmea = edited_df
                
                # Calculate RPN
                if all(col in edited_df.columns for col in ['Severity', 'Occurrence', 'Detection']):
                    edited_df['RPN'] = edited_df['Severity'] * edited_df['Occurrence'] * edited_df['Detection']
                    st.session_state.fmea_data = edited_df
                
                st.success("✅ FMEA updated!")
    
    def show_manual_entry_form(self):
        """Display manual entry form"""
        with st.form("manual_entry_form"):
            st.subheader("➕ Add Manual FMEA Entry")
            
            col1, col2 = st.columns(2)
            
            with col1:
                component = st.text_input("Component/System")
                failure_mode = st.text_input("Failure Mode")
                effect = st.text_area("Effect of Failure")
            
            with col2:
                cause = st.text_area("Potential Cause")
                detection = st.text_area("Detection Method")
                
                sev = st.slider("Severity", 1, 10, 5)
                occ = st.slider("Occurrence", 1, 10, 5)
                det = st.slider("Detection", 1, 10, 5)
            
            if st.form_submit_button("Add Entry"):
                new_entry = {
                    'Component': component,
                    'Failure Mode': failure_mode,
                    'Effect': effect,
                    'Cause': cause,
                    'Detection Method': detection,
                    'Severity': sev,
                    'Occurrence': occ,
                    'Detection': det,
                    'RPN': sev * occ * det
                }
                
                # Add to existing FMEA data
                new_df = pd.DataFrame([new_entry])
                st.session_state.fmea_data = pd.concat([st.session_state.fmea_data, new_df], ignore_index=True)
                st.success("✅ Manual entry added!")
                st.rerun()
    
    def chat_assistant_page(self):
        """Chat assistant page for FMEA refinement"""
        st.header("💬 Chat Assistant")
        
        if st.session_state.fmea_data.empty:
            st.warning("⚠️ Please generate an FMEA first!")
            return
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the FMEA or request modifications..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Get AI response
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.coordinator.chat_with_fmea(prompt, st.session_state.fmea_data)
                    st.write(response["message"])
                    
                    # If there are FMEA updates, apply them
                    if "updated_fmea" in response:
                        st.session_state.fmea_data = response["updated_fmea"]
                        st.success("✅ FMEA updated based on your request!")
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze High Risk Items"):
                high_risk_analysis = self.coordinator.analyze_high_risk_items(st.session_state.fmea_data)
                st.info(high_risk_analysis)
        
        with col2:
            if st.button("💡 Suggest Improvements"):
                suggestions = self.coordinator.suggest_improvements(st.session_state.fmea_data)
                st.info(suggestions)
        
        with col3:
            if st.button("🔄 Validate FMEA"):
                validation = self.coordinator.validate_fmea(st.session_state.fmea_data)
                st.info(validation)
    
    def risk_dashboard_page(self):
        """Risk dashboard visualization page"""
        st.header("📊 Risk Dashboard")
        
        if st.session_state.fmea_data.empty:
            st.warning("⚠️ No FMEA data available for visualization!")
            return
        
        # Create dashboard
        dashboard = VisualDashboard()
        dashboard.render(st.session_state.fmea_data)
    
    def report_export_page(self):
        """Report export page"""
        st.header("📋 Export FMEA Report")
        
        if st.session_state.fmea_data.empty:
            st.warning("⚠️ No FMEA data available for export!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current FMEA Summary")
            df = st.session_state.fmea_data
            
            st.metric("Total Entries", len(df))
            if 'RPN' in df.columns:
                st.metric("Average RPN", f"{df['RPN'].mean():.1f}")
                st.metric("Max RPN", df['RPN'].max())
                high_risk_count = len(df[df['RPN'] > 100])
                st.metric("High Risk Items (RPN > 100)", high_risk_count)
        
        with col2:
            st.subheader("📥 Export Options")
            
            export_format = st.selectbox(
                "Select Export Format:",
                ["Excel (.xlsx)", "CSV (.csv)", "PDF Report"]
            )
            
            include_charts = st.checkbox("Include Risk Charts", True)
            include_summary = st.checkbox("Include Executive Summary", True)
            
            if st.button("📤 Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    report_generator = ReportGeneratorAgent()
                    
                    if export_format == "Excel (.xlsx)":
                        file_data = report_generator.export_excel(
                            st.session_state.fmea_data,
                            include_charts=include_charts,
                            include_summary=include_summary
                        )
                        st.download_button(
                            "📥 Download Excel Report",
                            data=file_data,
                            file_name=f"FMEA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == "CSV (.csv)":
                        csv_data = st.session_state.fmea_data.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV Report",
                            data=csv_data,
                            file_name=f"FMEA_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    st.success("✅ Report generated successfully!")

def main():
    """Main entry point"""
    app = FMEASystem()
    app.run()

if __name__ == "__main__":
    main()