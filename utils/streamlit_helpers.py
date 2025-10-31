"""
Streamlit app configuration and state management
"""

import streamlit as st
import time
from datetime import datetime

class StreamlitState:
    """Manages Streamlit session state and progress tracking"""
    
    @staticmethod
    def initialize():
        """Initialize session state variables"""
        if 'process_start_time' not in st.session_state:
            st.session_state.process_start_time = time.time()
            st.session_state.last_update = time.time()
            st.session_state.training_in_progress = False
            st.session_state.progress_bar = None
            st.session_state.status_text = None
            st.session_state.error_text = None
    
    @staticmethod
    def create_containers():
        """Create placeholder containers for progress tracking"""
        if not st.session_state.progress_bar:
            st.session_state.progress_bar = st.progress(0)
        if not st.session_state.status_text:
            st.session_state.status_text = st.empty()
        if not st.session_state.error_text:
            st.session_state.error_text = st.empty()
    
    @staticmethod
    def update_progress(message: str, progress: float = None):
        """Update progress display"""
        current_time = time.time()
        elapsed = current_time - st.session_state.process_start_time
        
        if progress is not None and st.session_state.progress_bar:
            st.session_state.progress_bar.progress(progress)
        
        if st.session_state.status_text:
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][
                int(current_time * 10) % 10
            ]
            st.session_state.status_text.markdown(
                f"{spinner} **{message}** (elapsed: {elapsed:.1f}s)"
            )
        
        # Force UI update
        time.sleep(0.1)
    
    @staticmethod
    def show_error(error: Exception):
        """Display error message"""
        if st.session_state.error_text:
            st.session_state.error_text.error(f"❌ Error: {str(error)}")
    
    @staticmethod
    def reset():
        """Reset progress tracking state"""
        st.session_state.training_in_progress = False
        st.session_state.process_start_time = time.time()
        if st.session_state.progress_bar:
            st.session_state.progress_bar.empty()
        if st.session_state.status_text:
            st.session_state.status_text.empty()
        if st.session_state.error_text:
            st.session_state.error_text.empty()