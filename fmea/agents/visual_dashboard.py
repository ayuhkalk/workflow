# agents/visual_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

class VisualDashboard:
    """
    Handles visualization and dashboard creation for FMEA data
    Implements the VisualDashboard from the class diagram
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for different risk levels
        self.color_schemes = {
            'risk_levels': {
                'Low': '#2E8B57',      # Sea Green
                'Medium': '#FFA500',    # Orange  
                'High': '#FF6B6B',      # Red
                'Critical': '#8B0000'   # Dark Red
            },
            'components': px.colors.qualitative.Set3,
            'heatmap': 'RdYlBu_r'
        }
    
    def render(self, fmea_data: pd.DataFrame):
        """Main method to render the complete dashboard"""
        try:
            if fmea_data.empty:
                st.warning("⚠️ No FMEA data available for visualization")
                return
            
            # Dashboard header
            st.subheader("📊 Risk Analytics Dashboard")
            
            # Key metrics row
            self.render_key_metrics(fmea_data)
            
            # Main visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_rpn_distribution(fmea_data)
                self.render_component_risk_chart(fmea_data)
            
            with col2:
                self.render_risk_matrix(fmea_data)
                self.render_top_risks_table(fmea_data)
            
            # Additional analytics
            st.subheader("🔍 Detailed Analytics")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Risk Trends", "Component Analysis", "SOD Analysis", "Risk Heatmap"])
            
            with tab1:
                self.render_risk_trends(fmea_data)
            
            with tab2:
                self.render_component_analysis(fmea_data)
            
            with tab3:
                self.render_sod_analysis(fmea_data)
            
            with tab4:
                self.render_risk_heatmap(fmea_data)
                
        except Exception as e:
            self.logger.error(f"Dashboard rendering failed: {str(e)}")
            st.error(f"Dashboard rendering failed: {str(e)}")
    
    def render_key_metrics(self, fmea_data: pd.DataFrame):
        """Render key metrics cards"""
        try:
            # Calculate metrics
            total_entries = len(fmea_data)
            avg_rpn = fmea_data['RPN'].mean() if 'RPN' in fmea_data.columns else 0
            max_rpn = fmea_data['RPN'].max() if 'RPN' in fmea_data.columns else 0
            
            # Risk categorization
            if 'RPN' in fmea_data.columns:
                high_risk_count = len(fmea_data[fmea_data['RPN'] > 100])
                critical_risk_count = len(fmea_data[fmea_data['RPN'] > 200])
            else:
                high_risk_count = 0
                critical_risk_count = 0
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="📋 Total Entries",
                    value=total_entries
                )
            
            with col2:
                st.metric(
                    label="📊 Average RPN",
                    value=f"{avg_rpn:.1f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="⚠️ Max RPN",
                    value=int(max_rpn),
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="🔴 High Risk",
                    value=high_risk_count,
                    delta=f"{(high_risk_count/total_entries*100):.1f}%" if total_entries > 0 else "0%"
                )
            
            with col5:
                st.metric(
                    label="🚨 Critical Risk",
                    value=critical_risk_count,
                    delta=f"{(critical_risk_count/total_entries*100):.1f}%" if total_entries > 0 else "0%"
                )
                
        except Exception as e:
            self.logger.error(f"Key metrics rendering failed: {str(e)}")
    
    def render_rpn_distribution(self, fmea_data: pd.DataFrame):
        """Render RPN distribution chart"""
        try:
            st.subheader("📈 RPN Distribution")
            
            if 'RPN' not in fmea_data.columns:
                st.warning("RPN data not available")
                return
            
            # Create histogram
            fig = px.histogram(
                fmea_data, 
                x='RPN', 
                nbins=20,
                title="Distribution of Risk Priority Numbers",
                color_discrete_sequence=['#1f77b4']
            )
            
            # Add risk threshold lines
            fig.add_vline(x=50, line_dash="dash", line_color="green", 
                         annotation_text="Low/Medium Threshold")
            fig.add_vline(x=100, line_dash="dash", line_color="orange", 
                         annotation_text="Medium/High Threshold")
            fig.add_vline(x=200, line_dash="dash", line_color="red", 
                         annotation_text="High/Critical Threshold")
            
            fig.update_layout(
                xaxis_title="Risk Priority Number (RPN)",
                yaxis_title="Number of Failure Modes",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"RPN distribution rendering failed: {str(e)}")
    
    def render_risk_matrix(self, fmea_data: pd.DataFrame):
        """Render risk matrix (Severity vs Occurrence)"""
        try:
            st.subheader("🎯 Risk Matrix")
            
            required_cols = ['Severity', 'Occurrence']
            if not all(col in fmea_data.columns for col in required_cols):
                st.warning("Severity and Occurrence data not available")
                return
            
            # Create scatter plot
            fig = px.scatter(
                fmea_data,
                x='Occurrence',
                y='Severity',
                size='Detection' if 'Detection' in fmea_data.columns else None,
                color='RPN' if 'RPN' in fmea_data.columns else None,
                hover_data=['Component', 'Failure Mode'] if 'Component' in fmea_data.columns else None,
                title="Risk Matrix: Severity vs Occurrence",
                color_continuous_scale='Reds'
            )
            
            # Add risk zones
            fig.add_shape(
                type="rect", x0=0.5, y0=0.5, x1=5.5, y1=5.5,
                fillcolor="green", opacity=0.1, line_width=0
            )
            fig.add_shape(
                type="rect", x0=5.5, y0=0.5, x1=10.5, y1=5.5,
                fillcolor="yellow", opacity=0.1, line_width=0
            )
            fig.add_shape(
                type="rect", x0=0.5, y0=5.5, x1=5.5, y1=10.5,
                fillcolor="yellow", opacity=0.1, line_width=0
            )
            fig.add_shape(
                type="rect", x0=5.5, y0=5.5, x1=10.5, y1=10.5,
                fillcolor="red", opacity=0.1, line_width=0
            )
            
            fig.update_layout(
                xaxis_title="Occurrence Rating",
                yaxis_title="Severity Rating",
                xaxis=dict(range=[0.5, 10.5]),
                yaxis=dict(range=[0.5, 10.5]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Risk matrix rendering failed: {str(e)}")
    
    def render_component_risk_chart(self, fmea_data: pd.DataFrame):
        """Render component risk analysis chart"""
        try:
            st.subheader("🔧 Component Risk Analysis")
            
            if 'Component' not in fmea_data.columns or 'RPN' not in fmea_data.columns:
                st.warning("Component and RPN data not available")
                return
            
            # Calculate average RPN by component
            component_risk = fmea_data.groupby('Component').agg({
                'RPN': ['mean', 'max', 'count']
            }).round(1)
            
            component_risk.columns = ['Avg_RPN', 'Max_RPN', 'Count']
            component_risk = component_risk.reset_index()
            component_risk = component_risk.sort_values('Avg_RPN', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                component_risk,
                x='Avg_RPN',
                y='Component',
                color='Max_RPN',
                title="Average RPN by Component",
                color_continuous_scale='Reds',
                orientation='h'
            )
            
            fig.update_layout(
                xaxis_title="Average RPN",
                yaxis_title="Component",
                height=max(300, len(component_risk) * 25)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Component risk chart rendering failed: {str(e)}")
    
    def render_top_risks_table(self, fmea_data: pd.DataFrame):
        """Render top risks table"""
        try:
            st.subheader("🚨 Top Risk Items")
            
            if 'RPN' not in fmea_data.columns:
                st.warning("RPN data not available")
                return
            
            # Get top 10 highest RPN items
            top_risks = fmea_data.nlargest(10, 'RPN')
            
            # Display table
            display_columns = ['Component', 'Failure Mode', 'Effect', 'Severity', 'Occurrence', 'Detection', 'RPN']
            available_columns = [col for col in display_columns if col in top_risks.columns]
            
            # Style the dataframe
            def highlight_risk(val):
                if isinstance(val, (int, float)):
                    if val > 200:
                        return 'background-color: #ffcccc'  # Light red
                    elif val > 100:
                        return 'background-color: #ffe6cc'  # Light orange
                    elif val > 50:
                        return 'background-color: #ffffcc'  # Light yellow
                return ''
            
            styled_df = top_risks[available_columns].style.applymap(
                highlight_risk, subset=['RPN'] if 'RPN' in available_columns else []
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Top risks table rendering failed: {str(e)}")
    
    def render_risk_trends(self, fmea_data: pd.DataFrame):
        """Render risk trends analysis"""
        try:
            if 'Source' in fmea_data.columns:
                # Risk by source
                source_risk = fmea_data.groupby('Source')['RPN'].agg(['mean', 'count']).reset_index()
                
                fig = px.bar(
                    source_risk,
                    x='Source',
                    y='mean',
                    title="Average RPN by Data Source",
                    text='count'
                )
                fig.update_traces(texttemplate='Count: %{text}', textposition='outside')
                fig.update_layout(
                    xaxis_title="Data Source",
                    yaxis_title="Average RPN"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # RPN vs SOD scatter matrix
            if all(col in fmea_data.columns for col in ['Severity', 'Occurrence', 'Detection', 'RPN']):
                st.subheader("📊 SOD vs RPN Correlation")
                
                # Create correlation matrix
                corr_data = fmea_data[['Severity', 'Occurrence', 'Detection', 'RPN']].corr()
                
                fig = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix: S-O-D-RPN",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            self.logger.error(f"Risk trends rendering failed: {str(e)}")
    
    def render_component_analysis(self, fmea_data: pd.DataFrame):
        """Render detailed component analysis"""
        try:
            if 'Component' not in fmea_data.columns:
                st.warning("Component data not available")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Component failure mode count
                component_counts = fmea_data['Component'].value_counts()
                
                fig = px.pie(
                    values=component_counts.values,
                    names=component_counts.index,
                    title="Failure Modes by Component"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Component risk distribution
                if 'RPN' in fmea_data.columns:
                    fig = px.box(
                        fmea_data,
                        x='Component',
                        y='RPN',
                        title="RPN Distribution by Component"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed component table
            st.subheader("📋 Component Summary")
            if 'RPN' in fmea_data.columns:
                component_summary = fmea_data.groupby('Component').agg({
                    'RPN': ['count', 'mean', 'max', 'min'],
                    'Severity': 'mean',
                    'Occurrence': 'mean',
                    'Detection': 'mean'
                }).round(2)
                
                component_summary.columns = [
                    'Failure_Modes', 'Avg_RPN', 'Max_RPN', 'Min_RPN',
                    'Avg_Severity', 'Avg_Occurrence', 'Avg_Detection'
                ]
                
                st.dataframe(component_summary, use_container_width=True)
                
        except Exception as e:
            self.logger.error(f"Component analysis rendering failed: {str(e)}")
    
    def render_sod_analysis(self, fmea_data: pd.DataFrame):
        """Render Severity, Occurrence, Detection analysis"""
        try:
            required_cols = ['Severity', 'Occurrence', 'Detection']
            if not all(col in fmea_data.columns for col in required_cols):
                st.warning("SOD data not available")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                # SOD distributions
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Severity Distribution', 'Occurrence Distribution', 'Detection Distribution'),
                    vertical_spacing=0.1
                )
                
                # Severity
                fig.add_trace(
                    go.Histogram(x=fmea_data['Severity'], name='Severity', nbinsx=10),
                    row=1, col=1
                )
                
                # Occurrence  
                fig.add_trace(
                    go.Histogram(x=fmea_data['Occurrence'], name='Occurrence', nbinsx=10),
                    row=2, col=1
                )
                
                # Detection
                fig.add_trace(
                    go.Histogram(x=fmea_data['Detection'], name='Detection', nbinsx=10),
                    row=3, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # SOD averages by component
                if 'Component' in fmea_data.columns:
                    sod_by_component = fmea_data.groupby('Component')[required_cols].mean()
                    
                    fig = px.bar(
                        sod_by_component.reset_index(),
                        x='Component',
                        y=['Severity', 'Occurrence', 'Detection'],
                        title="Average SOD Ratings by Component",
                        barmode='group'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # SOD statistics
            st.subheader("📊 SOD Statistics")
            sod_stats = fmea_data[required_cols].describe()
            st.dataframe(sod_stats, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"SOD analysis rendering failed: {str(e)}")
    
    def render_risk_heatmap(self, fmea_data: pd.DataFrame):
        """Render risk heatmap"""
        try:
            if not all(col in fmea_data.columns for col in ['Severity', 'Occurrence']):
                st.warning("Severity and Occurrence data required for heatmap")
                return
            
            # Create heatmap data
            heatmap_data = fmea_data.groupby(['Severity', 'Occurrence']).size().reset_index(name='Count')
            
            # Create pivot table for heatmap
            pivot_table = heatmap_data.pivot(index='Severity', columns='Occurrence', values='Count')
            pivot_table = pivot_table.fillna(0)
            
            # Create heatmap
            fig = px.imshow(
                pivot_table,
                labels=dict(x="Occurrence", y="Severity", color="Count"),
                x=pivot_table.columns,
                y=pivot_table.index,
                title="Risk Heatmap: Severity vs Occurrence",
                color_continuous_scale='Reds',
                text_auto=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk zone analysis
            st.subheader("🎯 Risk Zone Analysis")
            
            # Calculate risk zones
            def get_risk_zone(severity, occurrence):
                if severity <= 5 and occurrence <= 5:
                    return 'Low Risk'
                elif severity <= 7 and occurrence <= 7:
                    return 'Medium Risk'
                elif severity <= 9 or occurrence <= 9:
                    return 'High Risk'
                else:
                    return 'Critical Risk'
            
            fmea_data['Risk_Zone'] = fmea_data.apply(
                lambda row: get_risk_zone(row['Severity'], row['Occurrence']), axis=1
            )
            
            zone_counts = fmea_data['Risk_Zone'].value_counts()
            
            # Display zone distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=zone_counts.values,
                    names=zone_counts.index,
                    title="Risk Zone Distribution",
                    color_discrete_map=self.color_schemes['risk_levels']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Risk Zone Breakdown:**")
                for zone, count in zone_counts.items():
                    percentage = (count / len(fmea_data)) * 100
                    st.write(f"• {zone}: {count} items ({percentage:.1f}%)")
                    
        except Exception as e:
            self.logger.error(f"Risk heatmap rendering failed: {str(e)}")
    
    def highlight_top_risks(self, fmea_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify and highlight top risk items"""
        try:
            if 'RPN' not in fmea_data.columns:
                return []
            
            # Get top 5 risks
            top_risks = fmea_data.nlargest(5, 'RPN')
            
            risk_highlights = []
            for _, row in top_risks.iterrows():
                risk_highlights.append({
                    'component': row.get('Component', 'Unknown'),
                    'failure_mode': row.get('Failure Mode', 'Unknown'),
                    'rpn': row.get('RPN', 0),
                    'severity': row.get('Severity', 0),
                    'occurrence': row.get('Occurrence', 0),
                    'detection': row.get('Detection', 0),
                    'risk_level': self._categorize_risk(row.get('RPN', 0))
                })
            
            return risk_highlights
            
        except Exception as e:
            self.logger.error(f"Risk highlighting failed: {str(e)}")
            return []
    
    def update_view(self, fmea_data: pd.DataFrame, filter_criteria: Dict = None):
        """Update dashboard view based on filter criteria"""
        try:
            filtered_data = fmea_data.copy()
            
            if filter_criteria:
                # Apply filters
                if 'min_rpn' in filter_criteria:
                    filtered_data = filtered_data[filtered_data['RPN'] >= filter_criteria['min_rpn']]
                
                if 'components' in filter_criteria:
                    filtered_data = filtered_data[filtered_data['Component'].isin(filter_criteria['components'])]
                
                if 'risk_level' in filter_criteria:
                    risk_level = filter_criteria['risk_level']
                    if risk_level == 'High':
                        filtered_data = filtered_data[filtered_data['RPN'] > 100]
                    elif risk_level == 'Critical':
                        filtered_data = filtered_data[filtered_data['RPN'] > 200]
            
            # Re-render with filtered data
            self.render(filtered_data)
            
        except Exception as e:
            self.logger.error(f"Dashboard update failed: {str(e)}")
    
    def _categorize_risk(self, rpn: float) -> str:
        """Categorize risk level based on RPN"""
        if rpn > 200:
            return 'Critical'
        elif rpn > 100:
            return 'High'
        elif rpn > 50:
            return 'Medium'
        else:
            return 'Low'
    
    def export_dashboard_data(self, fmea_data: pd.DataFrame) -> Dict[str, Any]:
        """Export dashboard data for reporting"""
        try:
            dashboard_data = {
                'summary_stats': {
                    'total_entries': len(fmea_data),
                    'avg_rpn': fmea_data['RPN'].mean() if 'RPN' in fmea_data.columns else 0,
                    'max_rpn': fmea_data['RPN'].max() if 'RPN' in fmea_data.columns else 0,
                    'high_risk_count': len(fmea_data[fmea_data['RPN'] > 100]) if 'RPN' in fmea_data.columns else 0
                },
                'top_risks': self.highlight_top_risks(fmea_data),
                'component_analysis': {},
                'timestamp': pd.Timestamp.now()
            }
            
            # Component analysis
            if 'Component' in fmea_data.columns and 'RPN' in fmea_data.columns:
                component_stats = fmea_data.groupby('Component')['RPN'].agg(['count', 'mean', 'max']).to_dict()
                dashboard_data['component_analysis'] = component_stats
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data export failed: {str(e)}")
            return {}
    
    def get_dashboard_insights(self, fmea_data: pd.DataFrame) -> List[str]:
        """Generate insights from dashboard data"""
        insights = []
        
        try:
            if fmea_data.empty:
                return ["No data available for analysis"]
            
            # RPN insights
            if 'RPN' in fmea_data.columns:
                avg_rpn = fmea_data['RPN'].mean()
                high_risk_count = len(fmea_data[fmea_data['RPN'] > 100])
                high_risk_pct = (high_risk_count / len(fmea_data)) * 100
                
                if avg_rpn > 100:
                    insights.append(f"⚠️ High average RPN ({avg_rpn:.1f}) indicates significant risk exposure")
                
                if high_risk_pct > 20:
                    insights.append(f"🚨 {high_risk_pct:.1f}% of items are high risk - immediate action needed")
                elif high_risk_pct > 10:
                    insights.append(f"⚠️ {high_risk_pct:.1f}% of items are high risk - review recommended")
                else:
                    insights.append(f"✅ Only {high_risk_pct:.1f}% of items are high risk - good risk profile")
            
            # Component insights
            if 'Component' in fmea_data.columns:
                component_counts = fmea_data['Component'].value_counts()
                top_component = component_counts.index[0]
                top_count = component_counts.iloc[0]
                
                insights.append(f"🔧 {top_component} has the most failure modes ({top_count})")
                
                if 'RPN' in fmea_data.columns:
                    component_risk = fmea_data.groupby('Component')['RPN'].mean().sort_values(ascending=False)
                    riskiest_component = component_risk.index[0]
                    risk_value = component_risk.iloc[0]
                    
                    insights.append(f"⚠️ {riskiest_component} has highest average risk (RPN: {risk_value:.1f})")
            
            # SOD insights
            sod_cols = ['Severity', 'Occurrence', 'Detection']
            available_sod = [col for col in sod_cols if col in fmea_data.columns]
            
            for col in available_sod:
                avg_rating = fmea_data[col].mean()
                if avg_rating > 7:
                    insights.append(f"📊 High average {col.lower()} rating ({avg_rating:.1f}) - focus area for improvement")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {str(e)}")
            return [f"Insight generation failed: {str(e)}"]