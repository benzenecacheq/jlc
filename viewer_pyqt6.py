#!/usr/bin/env python3
"""
PyQt6 GUI Viewer for Lumber List Matching Results

A modern GUI application with interactive SKU selection for multiple matches.
Features a table view with embedded listboxes for items with multiple SKU matches.
"""

import sys
import csv
import os
import subprocess
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QCheckBox, QHeaderView, QMessageBox, 
    QFileDialog, QStatusBar, QSplitter, QTextEdit, QFrame, QMenuBar,
    QMenu, QToolBar, QDialog, QListWidget, QListWidgetItem, QFormLayout,
    QGroupBox, QScrollArea, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QAction, QKeyEvent

class SKUSelectionDialog(QDialog):
    """Dialog for selecting SKU from multiple matches"""
    
    def __init__(self, matches: List[Dict], parent=None):
        super().__init__(parent)
        self.matches = matches
        self.selected_match = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Select SKU")
        self.setModal(True)
        self.resize(500, 300)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Multiple matches found. Please select the correct SKU:")
        layout.addWidget(instructions)
        
        # List widget for SKU selection
        self.list_widget = QListWidget()
        
        # Add "No matches" option first
        no_match_item = QListWidgetItem("❌ No matches (clear selection)")
        no_match_item.setData(Qt.ItemDataRole.UserRole, None)
        self.list_widget.addItem(no_match_item)
        
        # Add actual matches
        for i, match in enumerate(self.matches):
            confidence_symbol = self.get_confidence_symbol(match['confidence'])
            item_text = f"{confidence_symbol} {match['part_number']} - {match['description'][:50]}{'...' if len(match['description']) > 50 else ''}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, match)
            self.list_widget.addItem(item)
        
        # Select first item by default
        self.list_widget.setCurrentRow(0)
        
        # Connect double-click to accept selection
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
            
        layout.addWidget(self.list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_selection)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def get_confidence_symbol(self, confidence_str):
        """Get confidence symbol based on confidence value"""
        try:
            if not confidence_str or confidence_str == '':
                return "⚠"
            conf_val = float(confidence_str)
            if conf_val >= 0.8:
                return "✓"
            elif conf_val >= 0.5:
                return "○"
            else:
                return "⚠"
        except (ValueError, TypeError, AttributeError):
            return "⚠"
            
    def accept_selection(self):
        """Accept the selected SKU"""
        current_item = self.list_widget.currentItem()
        if current_item:
            self.selected_match = current_item.data(Qt.ItemDataRole.UserRole)
            self.accept()
        else:
            self.reject()

class DocumentProcessor(QObject):
    """Worker thread for processing documents"""
    finished = pyqtSignal(object, object)  # Signal with databases and scanned_items
    error = pyqtSignal(str)
    notify = pyqtSignal(str)  # Signal for status updates
    
    def __init__(self, pdf_file, api_key, database_files, output_dir, processing_dialog):
        super().__init__()
        self.pdf_file = pdf_file
        self.api_key = api_key
        self.database_files = database_files
        self.output_dir = output_dir
        self.processing_dialog = processing_dialog
        self.error_occurred = False
    
    def notify_function(self, message):
        """Notify function that updates the processing dialog"""
        self.notify.emit(message)
    
    def error_function(self, error_message):
        """Error function that shows modal error dialog"""
        self.error_occurred = True
        self.error.emit(error_message)
    
    def run(self):
        """Run the document processing in a separate thread"""
        try:
            from pdf2parts import run_matcher
            from pathlib import Path
            
            # Ensure output_dir is a Path object
            output_dir_path = Path(self.output_dir)
            
            # Run the matcher with notify and error functions
            result = run_matcher(
                document=self.pdf_file,
                api_key=self.api_key,
                database_names=self.database_files,
                training_data=None,
                use_ai_matching=False,
                output_dir=output_dir_path,
                debug=False,
                notify_func=self.notify_function,
                error_func=self.error_function
            )
            
            # Check if an error occurred (run_matcher returns None, None on error)
            if result is None or self.error_occurred:
                return  # Error was already handled by error_function
            
            # Extract databases and scanned_items from the result
            databases, scanned_items = result
            self.finished.emit(databases, scanned_items)
            
        except Exception as e:
            self.error.emit(str(e))

class ProcessingDialog(QDialog):
    """Dialog that shows processing status without OK button"""
    
    def __init__(self, pdf_file, api_key, database_files, output_dir, parent=None):
        super().__init__(parent)
        self.pdf_file = pdf_file
        self.api_key = api_key
        self.database_files = database_files
        self.output_dir = output_dir
        self.init_ui()
        self.start_processing()
        
    def init_ui(self):
        """Initialize the processing dialog UI"""
        self.setWindowTitle("Processing Document")
        self.setModal(True)
        self.setFixedSize(450, 220)
        
        layout = QVBoxLayout(self)
        
        # Processing message
        self.status_label = QLabel("Processing document...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumHeight(60)  # Allow for 3 lines max
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Details
        details_text = f"PDF: {Path(self.pdf_file).name}\n"
        details_text += f"Output: {Path(self.output_dir).name}\n"
        details_text += f"Databases: {len(self.database_files)}"
        
        self.details_label = QLabel(details_text)
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        # Cancel button (optional)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_btn)
        
    def start_processing(self):
        """Start the document processing in a separate thread"""
        # Create worker thread
        self.thread = QThread()
        self.worker = DocumentProcessor(
            self.pdf_file, self.api_key, self.database_files, self.output_dir, self
        )
        
        # Move worker to thread
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error.connect(self.processing_error)
        self.worker.notify.connect(self.update_status)  # Connect notify signal
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Start thread
        self.thread.start()
        
    def cancel_processing(self):
        """Cancel the processing"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
        self.reject()
        
    def processing_finished(self, databases, scanned_items):
        """Handle successful processing completion"""
        self.status_label.setText("Processing completed successfully!")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.cancel_btn.setText("Close")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.accept)
        
        # Show summary dialog and close immediately after
        self.show_summary_dialog(databases, scanned_items)
        
    def update_status(self, message):
        """Update the status label with a new message"""
        self.status_label.setText(message)
    
    def show_summary_dialog(self, databases, scanned_items):
        """Show a summary dialog with processing results"""
        # Calculate summary statistics
        matched_items = sum(1 for item in scanned_items if item.matches)
        total_items = len(scanned_items)
        match_rate = (matched_items / total_items * 100) if total_items > 0 else 0
        
        # Create summary message
        summary_text = f"SUMMARY:\n\n"
        summary_text += f"Items scanned: {total_items}\n"
        summary_text += f"Items matched: {matched_items}\n"
        summary_text += f"Match rate: {match_rate:.1f}%\n"
        summary_text += f"Databases searched: {len(databases)}\n"
        summary_text += f"Output files saved to: {Path(self.output_dir).name}\n"
        summary_text += f"  Report: report.txt\n"
        summary_text += f"  CSV: matches.csv"
        
        # Show summary dialog
        summary_dialog = QMessageBox(self)
        summary_dialog.setWindowTitle("Processing Complete")
        summary_dialog.setText(summary_text)
        summary_dialog.setIcon(QMessageBox.Icon.Information)
        summary_dialog.exec()
        
        # Auto-load the results
        self.auto_load_results()
        
        # Close this dialog immediately after summary is closed
        self.accept()
    
    def auto_load_results(self):
        """Automatically load the generated CSV results"""
        try:
            # Look for the matches.csv file in the output directory
            matches_csv = Path(self.output_dir) / "matches.csv"
            if matches_csv.exists():
                # Find the corresponding database file
                # Look for *_fixed.csv files in the output directory
                fixed_csv_files = list(Path(self.output_dir).glob("*_fixed.csv"))
                if fixed_csv_files:
                    database_file = fixed_csv_files[0]  # Use the first one found
                else:
                    # Fallback to looking for skulist_fixed.csv
                    database_file = Path(self.output_dir) / "skulist_fixed.csv"
                    if not database_file.exists():
                        database_file = None
                
                if database_file and database_file.exists():
                    # Load the results
                    self.parent().csv_file = str(matches_csv)
                    self.parent().database_file = str(database_file)
                    self.parent().load_data()
                    
                    # Save the loaded results to config
                    self.parent().save_settings()
                    
        except Exception as e:
            print(f"Error auto-loading results: {e}")
    
    def processing_error(self, error_message):
        """Handle processing error"""
        self.status_label.setText("Processing failed!")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.cancel_btn.setText("Close")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.reject)
        
        # Show error details
        QMessageBox.critical(self, "Processing Error", 
                           f"Error processing document:\n\n{error_message}")

class ProcessDocumentDialog(QDialog):
    """Dialog for processing documents with PDF2Parts"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_key = ""
        self.pdf_file = ""
        self.database_files = []
        self.parent_gui = parent
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Process Document")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # API Key section
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)
        
        self.api_key_input = QLineEdit()
        # Preload with environment variable
        self.api_key_input.setText(os.getenv('ANTHROPIC_API_KEY', ''))
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addRow("API Key:", self.api_key_input)
        
        layout.addWidget(api_group)
        
        # PDF File section
        pdf_group = QGroupBox("Document")
        pdf_layout = QFormLayout(pdf_group)
        
        pdf_file_layout = QHBoxLayout()
        self.pdf_file_input = QLineEdit()
        self.pdf_file_input.setReadOnly(True)
        pdf_file_layout.addWidget(self.pdf_file_input)
        
        browse_pdf_btn = QPushButton("Browse...")
        browse_pdf_btn.clicked.connect(self.browse_pdf_file)
        pdf_file_layout.addWidget(browse_pdf_btn)
        
        pdf_layout.addRow("PDF File:", pdf_file_layout)
        layout.addWidget(pdf_group)
        
        # Database Files section
        db_group = QGroupBox("Part Databases")
        db_layout = QVBoxLayout(db_group)
        
        # Add/Remove database files
        db_controls_layout = QHBoxLayout()
        add_db_btn = QPushButton("Add Database...")
        add_db_btn.clicked.connect(self.add_database_file)
        db_controls_layout.addWidget(add_db_btn)
        
        remove_db_btn = QPushButton("Remove Selected")
        remove_db_btn.clicked.connect(self.remove_database_file)
        db_controls_layout.addWidget(remove_db_btn)
        
        db_layout.addLayout(db_controls_layout)
        
        # Database list
        self.database_list = QListWidget()
        db_layout.addWidget(self.database_list)
        
        # Preload with skulist.csv from executable directory
        self.load_default_database()
        
        layout.addWidget(db_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        process_btn = QPushButton("Process Document")
        process_btn.clicked.connect(self.process_document)
        process_btn.setDefault(True)
        button_layout.addWidget(process_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def load_settings(self):
        """Load settings from parent GUI"""
        try:
            if not self.parent_gui:
                return
                
            # Load API key
            if hasattr(self.parent_gui, 'last_api_key') and self.parent_gui.last_api_key:
                self.api_key_input.setText(self.parent_gui.last_api_key)
            
            # Load PDF directory
            if hasattr(self.parent_gui, 'last_pdf_dir') and self.parent_gui.last_pdf_dir:
                self.last_pdf_dir = self.parent_gui.last_pdf_dir
            
            # Load database directory
            if hasattr(self.parent_gui, 'last_db_dir') and self.parent_gui.last_db_dir:
                self.last_db_dir = self.parent_gui.last_db_dir
            
            # Load database files
            if hasattr(self.parent_gui, 'last_database_files') and self.parent_gui.last_database_files:
                # Clear existing entries
                self.database_files.clear()
                self.database_list.clear()
                
                for db_file in self.parent_gui.last_database_files:
                    if db_file and os.path.exists(db_file):
                        self.database_files.append(db_file)
                        self.database_list.addItem(db_file)
                            
        except Exception as e:
            print(f"Error loading ProcessDocument settings: {e}")
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            if not self.parent_gui:
                return
                
            # Update parent GUI with current settings
            self.parent_gui.last_api_key = self.api_key_input.text().strip()
            # Save the directory, not the full file path
            if hasattr(self, 'last_pdf_dir') and self.last_pdf_dir:
                self.parent_gui.last_pdf_dir = self.last_pdf_dir
            elif self.pdf_file_input.text().strip():
                self.parent_gui.last_pdf_dir = str(Path(self.pdf_file_input.text().strip()).parent)
            # Save database directory
            if hasattr(self, 'last_db_dir') and self.last_db_dir:
                self.parent_gui.last_db_dir = self.last_db_dir
            self.parent_gui.last_database_files = self.database_files.copy()
            
            # Save to config file
            self.parent_gui.save_settings()
            
        except Exception as e:
            print(f"Error saving ProcessDocument settings: {e}")
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        self.save_settings()
        event.accept()
        
    def load_default_database(self):
        """Load the default skulist.csv from the executable directory"""
        try:
            # Get the directory where the executable is located
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                exe_dir = Path(sys.executable).parent
            else:
                # Running as script
                exe_dir = Path(__file__).parent
            
            default_db = exe_dir / "skulist.csv"
            if default_db.exists():
                self.database_files.append(str(default_db))
                self.database_list.addItem(str(default_db))
        except Exception as e:
            print(f"Could not load default database: {e}")
    
    def browse_pdf_file(self):
        """Browse for PDF file"""
        # Start from the last used directory if available
        start_dir = ""
        if hasattr(self, 'last_pdf_dir') and self.last_pdf_dir:
            start_dir = self.last_pdf_dir
        elif hasattr(self.parent_gui, 'last_pdf_dir') and self.parent_gui.last_pdf_dir:
            start_dir = self.parent_gui.last_pdf_dir
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", start_dir, "PDF Files (*.pdf);;All Files (*)"
        )
        if filename:
            self.pdf_file_input.setText(filename)
            self.pdf_file = filename
            # Save the directory (not the full path) for next time
            self.last_pdf_dir = str(Path(filename).parent)
    
    def add_database_file(self):
        """Add a database file to the list"""
        # Start from the last used directory if available
        start_dir = ""
        if hasattr(self, 'last_db_dir') and self.last_db_dir:
            start_dir = self.last_db_dir
        elif hasattr(self.parent_gui, 'last_db_dir') and self.parent_gui.last_db_dir:
            start_dir = self.parent_gui.last_db_dir
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Database File", start_dir, "CSV Files (*.csv);;All Files (*)"
        )
        if filename and filename not in self.database_files:
            self.database_files.append(filename)
            self.database_list.addItem(filename)
            # Save the directory for next time
            self.last_db_dir = str(Path(filename).parent)
    
    def remove_database_file(self):
        """Remove selected database file from the list"""
        current_row = self.database_list.currentRow()
        if current_row >= 0:
            item = self.database_list.takeItem(current_row)
            if item:
                filename = item.text()
                if filename in self.database_files:
                    self.database_files.remove(filename)
    
    def process_document(self):
        """Process the document using pdf2parts.py"""
        # Validate inputs
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Please enter an API key.")
            return
        
        pdf_file = self.pdf_file_input.text().strip()
        if not pdf_file:
            QMessageBox.warning(self, "Missing PDF File", "Please select a PDF file.")
            return
        
        if not os.path.exists(pdf_file):
            QMessageBox.warning(self, "File Not Found", f"PDF file not found: {pdf_file}")
            return
        
        if not self.database_files:
            QMessageBox.warning(self, "No Databases", "Please add at least one database file.")
            return
        
        # Validate database files exist
        for db_file in self.database_files:
            if not os.path.exists(db_file):
                QMessageBox.warning(self, "File Not Found", f"Database file not found: {db_file}")
                return
        
        # Store values
        self.api_key = api_key
        self.pdf_file = pdf_file
        
        # Create output directory
        pdf_path = Path(pdf_file)
        output_dir = pdf_path.parent / f"{pdf_path.stem}_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save settings before processing
        self.save_settings()
        
        # Close this dialog and open the processing dialog
        self.accept()
        
        # Open the processing dialog
        processing_dialog = ProcessingDialog(
            pdf_file, api_key, self.database_files, str(output_dir), self.parent()
        )
        processing_dialog.exec()

class FilterSpinBox(QSpinBox):
    """Custom SpinBox that only triggers filtering on specific events"""
    filter_requested = pyqtSignal()  # Emit when filtering should be applied
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        super().keyPressEvent(event)
        # Only trigger filtering on Enter, Up, Down, Page Up, Page Down
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, 
                          Qt.Key.Key_Up, Qt.Key.Key_Down,
                          Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self.filter_requested.emit()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events"""
        super().wheelEvent(event)
        # Trigger filtering on mouse wheel
        self.filter_requested.emit()

class SKUComboBox(QComboBox):
    """Custom ComboBox for SKU selection with confidence indicators"""
    sku_selected = pyqtSignal(dict)  # Emit selected match data
    
    def __init__(self, matches: List[Dict], parent=None):
        super().__init__(parent)
        self.matches = matches
        self.setup_combobox()
        
    def setup_combobox(self):
        """Setup the combobox with match data"""
        try:
            for i, match in enumerate(self.matches):
                confidence_symbol = self.get_confidence_symbol(match['confidence'])
                display_text = f"{confidence_symbol} {match['part_number']}"
                self.addItem(display_text, match)
            
            # Connect selection change
            self.currentIndexChanged.connect(self.on_selection_changed)
            
            # Set initial selection
            if self.matches:
                self.setCurrentIndex(0)
            
        except Exception as e:
            pass
    
    def get_confidence_symbol(self, confidence_str):
        """Get confidence symbol based on confidence value"""
        try:
            if not confidence_str or confidence_str == '':
                return "⚠"
            conf_val = float(confidence_str)
            if conf_val >= 0.8:
                return "✓"
            elif conf_val >= 0.5:
                return "○"
            else:
                return "⚠"
        except (ValueError, TypeError, AttributeError):
            return "⚠"
    
    def on_selection_changed(self, index):
        """Handle selection change"""
        try:
            if 0 <= index < len(self.matches):
                selected_match = self.matches[index]
                self.sku_selected.emit(selected_match)
        except Exception as e:
            pass

class LumberViewerGUI(QMainWindow):
    """Main GUI application for lumber list matching results"""
    
    def __init__(self):
        super().__init__()
        self.raw_data = []
        self.grouped_data = []
        self.filtered_data = []
        self.sku_comboboxes = {}  # Store combobox widgets
        self.csv_file = None
        self.database_file = None  # Will be set when CSV is loaded
        self.filter_timer = QTimer()  # Timer for delayed filtering
        self.filter_timer.setSingleShot(True)
        self.filter_timer.timeout.connect(self.apply_filters)
        self.row_item_data = {}  # Store item data for each row
        self.manual_overrides = {}  # Store manual SKU selections
        
        # Config file path
        self.config_file = Path.home() / ".jlc.ini"
        
        # Initialize settings attributes
        self.last_api_key = ""
        self.last_pdf_dir = ""
        self.last_db_dir = ""
        self.last_database_files = []
        
        self.init_ui()
        self.load_settings()  # Load settings after UI is initialized
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Lumber List Matching Results Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar with search
        self.create_toolbar()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create main content area
        self.create_table_widget(main_layout)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Initial status message
        self.status_bar.showMessage("Ready - Use Process Document to scan PDFs or File menu to load CSV data.")
        
        # Apply styling
        self.apply_styling()

        self.update_window_title()

    def update_window_title(self):
        """Update window title to include CSV filename if loaded"""
        base_title = "Lumber List Matching Results Viewer"
        if self.csv_file:
            csv_filename = Path(self.csv_file).name
            self.setWindowTitle(f"Lumber list {self.csv_file}")
        else:
            self.setWindowTitle(base_title)
    
    def save_settings(self):
        """Save current settings to config file"""
        try:
            config = configparser.ConfigParser()
            
            # Create sections
            config['ProcessDocument'] = {}
            config['LastResults'] = {}
            
            # Save ProcessDocument settings (will be populated by ProcessDocumentDialog)
            if hasattr(self, 'last_api_key') and self.last_api_key:
                config['ProcessDocument']['api_key'] = self.last_api_key
            if hasattr(self, 'last_pdf_dir') and self.last_pdf_dir:
                config['ProcessDocument']['pdf_dir'] = self.last_pdf_dir
            if hasattr(self, 'last_db_dir') and self.last_db_dir:
                config['ProcessDocument']['db_dir'] = self.last_db_dir
            if hasattr(self, 'last_database_files') and self.last_database_files:
                config['ProcessDocument']['database_files'] = '|'.join(self.last_database_files)
            
            # Save LastResults settings
            if self.csv_file:
                config['LastResults']['csv_file'] = str(self.csv_file)
            if self.database_file:
                config['LastResults']['database_file'] = str(self.database_file)
            
            # Write to file
            with open(self.config_file, 'w') as f:
                config.write(f)
                
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from config file"""
        try:
            if not self.config_file.exists():
                return
                
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Load ProcessDocument settings
            if 'ProcessDocument' in config:
                self.last_api_key = config.get('ProcessDocument', 'api_key', fallback='')
                self.last_pdf_dir = config.get('ProcessDocument', 'pdf_dir', fallback='')
                self.last_db_dir = config.get('ProcessDocument', 'db_dir', fallback='')
                db_files_str = config.get('ProcessDocument', 'database_files', fallback='')
                if db_files_str:
                    self.last_database_files = db_files_str.split('|')
                else:
                    self.last_database_files = []
                
            
            # Load LastResults settings
            if 'LastResults' in config:
                csv_file = config.get('LastResults', 'csv_file', fallback='')
                database_file = config.get('LastResults', 'database_file', fallback='')
                
                if csv_file and os.path.exists(csv_file):
                    self.csv_file = csv_file
                    if database_file and os.path.exists(database_file):
                        self.database_file = database_file
                        self.load_data()
                        
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        self.save_settings()
        event.accept()
        
    def delayed_apply_filters(self):
        """Apply filters with a delay to prevent crashes while typing"""
        self.filter_timer.stop()  # Stop any existing timer
        self.filter_timer.start(300)  # Wait 300ms before applying filters
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Process Document action (moved to top)
        process_doc_action = QAction('&Process Document...', self)
        process_doc_action.setShortcut('Ctrl+P')
        process_doc_action.setStatusTip('Process PDF document with PDF2Parts')
        process_doc_action.triggered.connect(self.process_document)
        file_menu.addAction(process_doc_action)
        
        file_menu.addSeparator()
        
        # Load CSV action
        load_csv_action = QAction('&Load CSV Results...', self)
        load_csv_action.setShortcut('Ctrl+O')
        load_csv_action.setStatusTip('Load CSV matching results file')
        load_csv_action.triggered.connect(self.load_csv_file)
        file_menu.addAction(load_csv_action)
        
        # Load Database action
        load_db_action = QAction('Load &Database...', self)
        load_db_action.setShortcut('Ctrl+D')
        load_db_action.setStatusTip('Load SKU database file')
        load_db_action.triggered.connect(self.load_database_file)
        file_menu.addAction(load_db_action)
        
        file_menu.addSeparator()
        
        # Export action
        export_action = QAction('&Export CSV...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.setStatusTip('Export filtered results to CSV')
        export_action.triggered.connect(self.export_to_csv)
        export_action.setEnabled(False)
        self.export_action = export_action  # Store reference for enabling/disabling
        file_menu.addAction(export_action)
        
        # Export to POS System action
        export_pos_action = QAction('Export to &POS System...', self)
        export_pos_action.setShortcut('Ctrl+R')
        export_pos_action.setStatusTip('Export to POS System format (SKU, Qty, Linear Feet)')
        export_pos_action.triggered.connect(self.export_to_pos_system)
        export_pos_action.setEnabled(False)
        self.export_pos_action = export_pos_action  # Store reference for enabling/disabling
        file_menu.addAction(export_pos_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Filter actions
        self.show_matches_action = QAction('Show Items with &Matches', self)
        self.show_matches_action.setCheckable(True)
        self.show_matches_action.setChecked(True)
        self.show_matches_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_matches_action)
        
        self.show_no_matches_action = QAction('Show Items with &No Matches', self)
        self.show_no_matches_action.setCheckable(True)
        self.show_no_matches_action.setChecked(True)
        self.show_no_matches_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_no_matches_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self):
        """Create the toolbar with search and filters"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Search
        toolbar.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search items...")
        self.search_input.setMaximumWidth(200)
        self.search_input.textChanged.connect(self.delayed_apply_filters)
        toolbar.addWidget(self.search_input)
        
        toolbar.addSeparator()
        
        # Confidence filter
        toolbar.addWidget(QLabel("Min Confidence:"))
        self.confidence_spin = FilterSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(0)
        self.confidence_spin.setSuffix("%")
        self.confidence_spin.setMaximumWidth(80)
        # Only apply filters on specific events (Enter, arrows, wheel)
        self.confidence_spin.filter_requested.connect(self.apply_filters)
        # Also apply when editing is finished (focus lost)
        self.confidence_spin.editingFinished.connect(self.apply_filters)
        toolbar.addWidget(self.confidence_spin)
        
        toolbar.addSeparator()
        
        # Clear filters button
        clear_filters_action = QAction('Clear Filters', self)
        clear_filters_action.setStatusTip('Clear all filters')
        clear_filters_action.triggered.connect(self.clear_filters)
        toolbar.addAction(clear_filters_action)
        
    def create_table_widget(self, parent):
        """Create the main table widget"""
        self.table = QTableWidget()
        parent.addWidget(self.table)
        
        # Set table properties
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        
        # Connect click and double-click events
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        
        # Set up columns
        self.columns = [
            "Item #", "Quantity", "Text", "Original", 
            "SKU", "Description", "Type", "Confidence"
        ]
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Item #
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # Quantity
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Text
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Original
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # SKU
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # Description
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)  # Type
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)  # Confidence
        
        self.table.setColumnWidth(0, 60)   # Item #
        self.table.setColumnWidth(1, 80)   # Quantity
        self.table.setColumnWidth(4, 120)  # SKU
        self.table.setColumnWidth(6, 100)  # Type
        self.table.setColumnWidth(7, 100)  # Confidence
        
    def apply_styling(self):
        """Apply custom styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                alternate-background-color: #f8f8f8;
            }
            QTableWidget::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QComboBox {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 2px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background-color: #3daee9;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2a9fd6;
            }
            QPushButton:pressed {
                background-color: #1e7bb8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
    def load_csv_file(self):
        """Load CSV file using file dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load CSV Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            self.csv_file = filename
            # Auto-detect database file in same directory
            csv_dir = Path(filename).parent
            self.database_file = csv_dir / "skulist_fixed.csv"
            self.load_data()
            
    def load_database_file(self):
        """Load database file using file dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load SKU Database", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            self.database_file = filename
            self.load_data()
    
    def process_document(self):
        """Open the process document dialog"""
        dialog = ProcessDocumentDialog(self)
        dialog.exec()
            
    def load_data(self):
        """Load data from current CSV and database files"""
        if not self.csv_file or not self.database_file:
            QMessageBox.warning(self, "Missing Files", 
                              "Please load both CSV results and database files")
            return
            
        if not Path(self.csv_file).exists():
            QMessageBox.warning(self, "File Not Found", 
                              f"CSV file not found: {self.csv_file}")
            return
            
        if not Path(self.database_file).exists():
            QMessageBox.warning(self, "File Not Found", 
                              f"Database file not found: {self.database_file}")
            return
        
        try:
            # Clear previous data and overrides when loading new files
            self.manual_overrides.clear()
            self.row_item_data.clear()
            self.sku_comboboxes.clear()
            
            # Load database
            self.description_mapping, self.type_mapping, self.stocking_multiple_mapping = self.load_database(self.database_file)
            
            # Load and process CSV
            self.raw_data = self.process_csv_file(self.csv_file, self.description_mapping, self.type_mapping)
            self.group_data()
            self.apply_filters()
            self.update_display()
            self.update_window_title()  # Update window title with CSV filename
            
            self.export_action.setEnabled(True)
            self.export_pos_action.setEnabled(True)
            self.status_bar.showMessage(f"Loaded {len(self.raw_data)} rows, {len(self.grouped_data)} unique items")
            
            # Save the loaded files to config
            self.save_settings()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading files: {str(e)}")
            
    def load_database(self, database_file: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Load database file and create mappings"""
        part_to_description = {}
        part_to_type = {}
        part_to_stocking_multiple = {}
        
        with open(database_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                part_number = row.get('Item Number', '').strip()
                description = row.get('Item Description', '').strip()
                stocking_multiple = row.get('Stocking Multiple', '').strip()
                
                # Find type column (look for "Terms" in column name)
                item_type = ""
                for key, val in row.items():
                    if "Terms" in key:
                        item_type = val
                        break
                
                if part_number and description:
                    part_to_description[part_number] = description
                    part_to_type[part_number] = item_type
                    part_to_stocking_multiple[part_number] = stocking_multiple
                    
        return part_to_description, part_to_type, part_to_stocking_multiple
        
    def process_csv_file(self, csv_file: str, description_mapping: Dict[str, str], 
                        type_mapping: Dict[str, str]) -> List[Dict]:
        """Process the CSV file and combine with database information"""
        results = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Store the original column names for export
            self.original_csv_columns = reader.fieldnames
            
            for row in reader:
                # Extract data from CSV
                item_number = row.get('Item_Number', '').strip()
                quantity = row.get('Quantity', '').strip()
                processed_text = row.get('Description', '').strip()
                original_text = row.get('Original_Text', '').strip()
                part_number = row.get('Part_Number', '').strip()
                confidence = row.get('Confidence', '').strip()
                
                # Get description from database if part number exists
                item_description = ""
                item_type = ""
                if part_number and part_number in description_mapping:
                    item_description = description_mapping[part_number]
                    item_type = type_mapping[part_number]
                
                # Create result row - preserve all original columns plus add computed ones
                result_row = dict(row)  # Start with all original columns
                
                # Add/update specific fields we need for processing
                result_row.update({
                    'item_number': item_number,
                    'quantity': quantity,
                    'processed_text': processed_text,
                    'original_text': original_text,
                    'part_number': part_number if part_number else '',
                    'item_description': item_description,
                    'item_type': item_type,
                    'confidence': confidence,
                })
                
                results.append(result_row)
        
        return results
        
    def group_data(self):
        """Group raw data by item number"""
        self.grouped_data = []
        current_item = None
        
        for row in self.raw_data:
            item_number = row['item_number']
            
            if current_item is None or current_item['item_number'] != item_number:
                # New item - preserve all original CSV columns
                current_item = dict(row)  # Start with all original data
                current_item['matches'] = []  # Add matches list
                self.grouped_data.append(current_item)
            
            # Add match to current item
            if row['part_number']:  # Only add if there's a match
                current_item['matches'].append({
                    'part_number': row['part_number'],
                    'description': row['item_description'],
                    'type': row['item_type'],
                    'confidence': row['confidence']
                })
                
    def apply_filters(self):
        """Apply current filters to the data"""
        try:
            # Safety check - don't filter if no data is loaded
            if not self.grouped_data:
                return
                
            search_text = self.search_input.text().lower()
            min_confidence = self.confidence_spin.value() / 100.0
            show_matches = self.show_matches_action.isChecked()
            show_no_matches = self.show_no_matches_action.isChecked()
            
            self.filtered_data = []
            
            for i, item in enumerate(self.grouped_data):
                
                # Check search filter
                if search_text:
                    searchable_text = f"{item['item_number']} {item['processed_text']} {item['original_text']}".lower()
                    if search_text not in searchable_text:
                        continue
                
                # Check match status filter
                has_matches = len(item['matches']) > 0
                if has_matches and not show_matches:
                    continue
                if not has_matches and not show_no_matches:
                    continue
                
                # Check confidence filter with error handling
                if has_matches:
                    try:
                        confidences = []
                        for match in item['matches']:
                            if match.get('confidence'):
                                try:
                                    conf_val = float(match['confidence'])
                                    confidences.append(conf_val)
                                except (ValueError, TypeError):
                                    continue
                        
                        if confidences:
                            max_confidence = max(confidences)
                            if max_confidence < min_confidence:
                                continue
                    except Exception as e:
                        # If there's an error, include the item anyway
                        pass
                self.filtered_data.append(item)
                
            self.update_display()
            
        except Exception as e:
            print(f"Error in apply_filters: {e}")
            # Fallback: show all data
            self.filtered_data = self.grouped_data.copy()
            self.update_display()
        
    def clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.confidence_spin.setValue(0)
        self.show_matches_action.setChecked(True)
        self.show_no_matches_action.setChecked(True)
        self.apply_filters()
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Lumber Viewer", 
                         "Lumber List Matching Results Viewer\n\n"
                         "A PyQt6 application for viewing and interacting with\n"
                         "lumber list matching results with interactive SKU selection.")
        
    def update_display(self):
        """Update the table display"""
        try:
            # Safety check - ensure table still exists
            if not hasattr(self, 'table') or self.table is None:
                return
                
            # Clear existing comboboxes
            for combobox in self.sku_comboboxes.values():
                try:
                    combobox.deleteLater()
                except Exception:
                    pass
            self.sku_comboboxes.clear()
            
            # Safety check
            if not self.filtered_data:
                self.table.setRowCount(0)
                self.status_bar.showMessage("No data to display")
                return
            
            # Set table rows
            try:
                # Limit the number of rows to prevent memory issues
                max_rows = min(len(self.filtered_data), 200)  # Limit to 200 rows
                if max_rows < len(self.filtered_data):
                    self.filtered_data = self.filtered_data[:max_rows]
                
                self.table.setRowCount(len(self.filtered_data))
            except Exception as e:
                print(f"Error setting table row count: {e}")
                return
            
            for row_idx, item in enumerate(self.filtered_data):
                # Safety check - ensure row index is within table bounds
                if row_idx >= self.table.rowCount():
                    break
                
                # Store item data for this row
                self.row_item_data[row_idx] = item
                    
                try:
                    # Basic item info
                    self.table.setItem(row_idx, 0, QTableWidgetItem(str(item['item_number'])))
                    self.table.setItem(row_idx, 1, QTableWidgetItem(str(item['quantity'])))
                    self.table.setItem(row_idx, 2, QTableWidgetItem(item['processed_text']))
                    self.table.setItem(row_idx, 3, QTableWidgetItem(item['original_text']))
                except Exception as e:
                    continue
                
                # Use the unified display method
                self.update_display_for_row(row_idx, item)
            
            # Update status
            try:
                status_msg = f"Showing {len(self.filtered_data)} of {len(self.grouped_data)} items"
                self.status_bar.showMessage(status_msg)
            except Exception:
                self.status_bar.showMessage("Display updated")
            
        except Exception as e:
            print(f"Error in update_display: {e}")
            self.status_bar.showMessage("Error updating display")
        
    def on_cell_clicked(self, row: int, column: int):
        """Handle single click on table cell"""
        if row in self.row_item_data:
            item = self.row_item_data[row]
            if len(item['matches']) > 1:
                # Multiple matches - open SKU selection dialog
                self.open_sku_dialog(row, item)
            elif len(item['matches']) == 1:
                # Single match - open dialog to allow changing
                self.open_sku_dialog(row, item)
            else:
                # No matches - no action needed
                pass
    
    def on_cell_double_clicked(self, row: int, column: int):
        """Handle double-click on table cell"""
        if row in self.row_item_data:
            item = self.row_item_data[row]
            if len(item['matches']) > 1 or len(item['matches']) == 1:
                # Multiple or single matches - open SKU selection dialog
                self.open_sku_dialog(row, item)
            else:
                # No matches - no action needed
                pass
    
    def open_sku_dialog(self, row: int, item: dict):
        """Open SKU selection dialog for an item"""
        dialog = SKUSelectionDialog(item['matches'], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.selected_match is None:
                # "No matches" selected - store as override
                self.manual_overrides[row] = None
                self.update_display_for_row(row, item)
            else:
                # Match selected - store as override
                self.manual_overrides[row] = dialog.selected_match
                self.update_display_for_row(row, item)
    
    def highlight_row(self, row: int):
        """Highlight a row to indicate manual override"""
        # Set background color for the entire row
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setBackground(QColor(255, 255, 200))  # Light yellow background
            else:
                # Create item if it doesn't exist to ensure highlighting
                new_item = QTableWidgetItem("")
                new_item.setBackground(QColor(255, 255, 200))
                self.table.setItem(row, col, new_item)
    
    def update_display_for_row(self, row: int, item: dict):
        """Update display for a single row"""
        if row not in self.row_item_data:
            return
            
        # Check if there's a manual override
        if row in self.manual_overrides:
            override = self.manual_overrides[row]
            if override is None:
                # "No matches" was selected - show bold text and clear fields
                sku_item = QTableWidgetItem("No matches")
                font = sku_item.font()
                font.setBold(True)
                sku_item.setFont(font)
                self.table.setItem(row, 4, sku_item)
                # Clear description and other fields using update_item_details
                self.update_item_details(row, None)
            else:
                # Show manual override with bold text (user made a choice)
                sku_item = QTableWidgetItem(f"🔧 {override['part_number']}")
                font = sku_item.font()
                font.setBold(True)
                sku_item.setFont(font)
                self.table.setItem(row, 4, sku_item)
                self.update_item_details(row, override)
            
            # Highlight the entire row for manual overrides AFTER updating all items
            self.highlight_row(row)
        elif len(item['matches']) > 1:
            # Multiple matches - show clickable indicator with bold text (needs user action)
            sku_item = QTableWidgetItem(f"📋 {len(item['matches'])} matches (click to select)")
            font = sku_item.font()
            font.setBold(True)
            sku_item.setFont(font)
            self.table.setItem(row, 4, sku_item)
            # Set initial values for the first match
            if item['matches']:
                self.update_item_details(row, item['matches'][0])
        elif len(item['matches']) == 1:
            # Single match - show normal text (no action needed)
            match = item['matches'][0]
            sku_item = QTableWidgetItem(match['part_number'])
            self.table.setItem(row, 4, sku_item)
            self.update_item_details(row, match)
        else:
            # No matches - show normal text (no action needed)
            sku_item = QTableWidgetItem("No matches")
            self.table.setItem(row, 4, sku_item)
            self.table.setItem(row, 5, QTableWidgetItem(""))
            self.table.setItem(row, 6, QTableWidgetItem(""))
            self.table.setItem(row, 7, QTableWidgetItem(""))
    
    def on_sku_selected(self, row_idx: int, match: Dict):
        """Handle SKU selection from combobox"""
        try:
            self.update_item_details(row_idx, match)
        except Exception as e:
            pass
        
    def update_item_details(self, row_idx: int, match: Dict):
        """Update the description, type, and confidence columns"""
        try:
            if match is None:
                # "No matches" selected - clear all fields
                self.table.setItem(row_idx, 5, QTableWidgetItem(""))
                self.table.setItem(row_idx, 6, QTableWidgetItem(""))
                self.table.setItem(row_idx, 7, QTableWidgetItem(""))
            else:
                # Normal match - update fields
                description = match.get('description', '')
                item_type = match.get('type', '')
                confidence = match.get('confidence', '')
                
                self.table.setItem(row_idx, 5, QTableWidgetItem(str(description)))
                self.table.setItem(row_idx, 6, QTableWidgetItem(str(item_type)))
                
                confidence_symbol = self.get_confidence_symbol(confidence)
                # Format confidence to 2 decimal places
                try:
                    confidence_float = float(confidence)
                    confidence_text = f"{confidence_symbol} {confidence_float:.2f}"
                except (ValueError, TypeError):
                    confidence_text = f"{confidence_symbol} {confidence}"
                
                self.table.setItem(row_idx, 7, QTableWidgetItem(confidence_text))
        except Exception as e:
            # Set safe defaults
            try:
                self.table.setItem(row_idx, 5, QTableWidgetItem(""))
                self.table.setItem(row_idx, 6, QTableWidgetItem(""))
                self.table.setItem(row_idx, 7, QTableWidgetItem("⚠"))
            except Exception as e2:
                pass
        
    def get_confidence_symbol(self, confidence_str):
        """Get confidence symbol based on confidence value"""
        try:
            if not confidence_str or confidence_str == '':
                return "⚠"
            conf_val = float(confidence_str)
            if conf_val >= 0.8:
                return "✓"
            elif conf_val >= 0.5:
                return "○"
            else:
                return "⚠"
        except (ValueError, TypeError, AttributeError):
            return "⚠"
            
    def export_to_csv(self):
        """Export filtered data to CSV with manual overrides"""
        if not self.filtered_data:
            QMessageBox.information(self, "No Data", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "filtered_results.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Use original CSV columns if available, otherwise fall back to display columns
                    if hasattr(self, 'original_csv_columns') and self.original_csv_columns:
                        # Create a mapping of original columns to their values
                        export_columns = list(self.original_csv_columns)
                        writer.writerow(export_columns)
                    else:
                        writer.writerow(self.columns)
                        export_columns = self.columns
                    
                    for i, item in enumerate(self.filtered_data):
                        # Check if there's a manual override for this row
                        if i in self.manual_overrides:
                            override = self.manual_overrides[i]
                            if override is None:
                                # "No matches" was selected - update the relevant fields
                                export_row = self.create_export_row(item, export_columns)
                                # Update Part_Number, Type, Confidence fields (keep original Description)
                                if 'Part_Number' in export_row:
                                    export_row['Part_Number'] = ""  # Empty, not "No matches"
                                if 'Type' in export_row:
                                    export_row['Type'] = ""
                                if 'Confidence' in export_row:
                                    export_row['Confidence'] = ""
                                # Note: Description field keeps original CSV value
                                
                                # Write the row
                                row_values = [export_row.get(col, '') for col in export_columns]
                                writer.writerow(row_values)
                            else:
                                # Specific match was selected - update the relevant fields
                                # Create a copy of the item with the override data
                                item_with_override = item.copy()
                                item_with_override['part_number'] = override['part_number']
                                item_with_override['item_description'] = override['description']
                                item_with_override['item_type'] = override['type']
                                item_with_override['confidence'] = override['confidence']
                                
                                # Pass the override data to create_export_row so it can set the correct database description
                                export_row = self.create_export_row(item_with_override, export_columns, override)
                                
                                # Write the row
                                row_values = [export_row.get(col, '') for col in export_columns]
                                writer.writerow(row_values)
                        elif len(item['matches']) > 0:
                            # No manual override - create one row for each match
                            for match in item['matches']:
                                # Create a copy of the item with the specific match data
                                item_with_match = item.copy()
                                item_with_match['part_number'] = match['part_number']
                                item_with_match['item_description'] = match['description']
                                item_with_match['item_type'] = match['type']
                                item_with_match['confidence'] = match['confidence']
                                
                                # Pass the match data to create_export_row so it can set the correct database description
                                export_row = self.create_export_row(item_with_match, export_columns, match)
                                
                                # Write the row
                                row_values = [export_row.get(col, '') for col in export_columns]
                                writer.writerow(row_values)
                        else:
                            # No matches at all
                            export_row = self.create_export_row(item, export_columns)
                            export_row['Part_Number'] = ""  # Empty, not "No matches"
                            # Note: Description field keeps original CSV value
                            
                            # Write the row
                            row_values = [export_row.get(col, '') for col in export_columns]
                            writer.writerow(row_values)
                            
                QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
    
    def create_export_row(self, item: dict, export_columns: list, match_data: dict = None) -> dict:
        """Create an export row with all original CSV data plus computed fields"""
        export_row = {}
        
        # Start with original CSV data if available
        # The grouped data structure has the original CSV data in the first match or we need to reconstruct it
        if any(key in item for key in export_columns):
            for col in export_columns:
                export_row[col] = item.get(col, '')
        else:
            # Fallback to basic mapping
            for col in export_columns:
                export_row[col] = ''
        
        # Ensure our computed fields are properly mapped
        if 'Item_Number' in export_columns:
            export_row['Item_Number'] = item.get('item_number', '')
        if 'Quantity' in export_columns:
            export_row['Quantity'] = item.get('quantity', '')
        if 'Description' in export_columns:
            # Description should be the original CSV "Description" field (processed_text)
            export_row['Description'] = item.get('processed_text', '')
        if 'Original_Text' in export_columns:
            export_row['Original_Text'] = item.get('original_text', '')
        
        # Set Part_Number - prioritize match_data over item data
        part_number = None
        if match_data and 'part_number' in match_data:
            part_number = match_data['part_number']
        else:
            part_number = item.get('part_number', '')
        
        if 'Part_Number' in export_columns:
            export_row['Part_Number'] = part_number
        if 'Confidence' in export_columns:
            export_row['Confidence'] = item.get('confidence', '')
        
        if 'Database_Description' in export_columns and part_number != item.get('Part_Number'):
            # look up the raw data for the description
            for r in self.raw_data:
                if r.get('Item_Number') == item['Item_Number'] and r.get('Part_Number') == part_number:
                    export_row['Database_Description'] = r.get('Database_Description')
            
        return export_row
    
    def export_to_pos_system(self):
        """Export filtered data to POS System format (SKU, Qty, Linear Feet)"""
        if not self.filtered_data:
            QMessageBox.information(self, "No Data", "No data to export")
            return
        
        # Generate filename based on input CSV name
        if self.csv_file:
            csv_path = Path(self.csv_file)
            filename = csv_path.parent / f"{csv_path.stem}_pos.csv"
        else:
            filename = "pos_export.csv"
        
        # Ask user to confirm the filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export to POS System", str(filename), "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    # writer.writerow(['SKU', 'Qty', 'Linear Feet'])
                    
                    for i, item in enumerate(self.filtered_data):
                        # Check if there's a manual override for this row
                        if i in self.manual_overrides:
                            override = self.manual_overrides[i]
                            if override is None:
                                # "No matches" was selected - skip this item
                                continue
                            else:
                                # Use the manual override
                                sku = override['part_number']
                                quantity = item.get('quantity', '')
                                stocking_multiple = self.stocking_multiple_mapping.get(sku, '')
                                
                                if stocking_multiple == 'LF':
                                    # For LF items: Qty = 1, Linear Feet = quantity
                                    writer.writerow([sku, '1', quantity])
                                else:
                                    # For other items: Qty = quantity, Linear Feet = empty
                                    writer.writerow([sku, quantity, ''])
                        elif len(item['matches']) > 0:
                            # No manual override - use the first match
                            match = item['matches'][0]
                            sku = match['part_number']
                            quantity = item.get('quantity', '')
                            stocking_multiple = self.stocking_multiple_mapping.get(sku, '')
                            
                            if stocking_multiple == 'LF':
                                # For LF items: Qty = 1, Linear Feet = quantity
                                writer.writerow([sku, '1', quantity])
                            else:
                                # For other items: Qty = quantity, Linear Feet = empty
                                writer.writerow([sku, quantity, ''])
                        # Skip items with no matches
                            
                QMessageBox.information(self, "Export Complete", f"POS data exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting POS data: {str(e)}")

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Lumber List Matching Results Viewer")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = LumberViewerGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
