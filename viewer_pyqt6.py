#!/usr/bin/env python3
"""
PyQt6 GUI Viewer for Lumber List Matching Results

A modern GUI application with interactive SKU selection for multiple matches.
Features a table view with embedded listboxes for items with multiple SKU matches.
"""

import sys
import csv
import os
import pdb
import re
import subprocess
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QCheckBox, QHeaderView, QMessageBox, 
    QFileDialog, QStatusBar, QSplitter, QTextEdit, QFrame, QMenuBar,
    QMenu, QToolBar, QDialog, QListWidget, QListWidgetItem, QFormLayout,
    QGroupBox, QScrollArea, QProgressBar, QStyledItemDelegate
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject, QModelIndex
from PyQt6.QtGui import QFont, QPalette, QColor, QAction, QKeyEvent, QPainter, QPen

from util import *

class DeletedItemDelegate(QStyledItemDelegate):
    """Custom delegate that draws a continuous line across deleted rows"""
    def __init__(self, parent=None, deleted_items=None):
        super().__init__(parent)
        self.deleted_items = deleted_items or set()
        self.row_item_data = {}
    
    def set_deleted_items(self, deleted_items):
        """Update the set of deleted items"""
        self.deleted_items = deleted_items
    
    def set_row_item_data(self, row_item_data):
        """Update the row item data mapping"""
        self.row_item_data = row_item_data
    
    def paint(self, painter, option, index):
        """Paint the item with a continuous line if deleted"""
        # First, do the standard painting
        super().paint(painter, option, index)
        
        # Check if this row represents a deleted item
        row = index.row()
        if row in self.row_item_data:
            item = self.row_item_data[row]
            item_number = item.get('item_number')
            if item_number and item_number in self.deleted_items:
                # Draw a line across this cell
                # Since all cells in the row are painted with the same delegate,
                # the lines will appear continuous across the entire row
                painter.save()
                pen = QPen(QColor(0, 0, 0), 2)  # Black pen, 2 pixels wide for visibility
                painter.setPen(pen)
                
                # Get the row height and calculate the middle
                row_height = option.rect.height()
                y = option.rect.top() + row_height // 2
                
                # Draw line from left edge to right edge of the cell
                # The line will appear continuous because all cells in the row are painted
                painter.drawLine(option.rect.left(), y, option.rect.right(), y)
                painter.restore()

class KeywordFileDialog(QDialog):
    """Dialog for selecting keyword file with text input and browse button"""
    
    def __init__(self, current_file="", parent=None):
        super().__init__(parent)
        self.current_file = current_file
        self.parent_gui = parent
        self.last_keyword_dir = ""
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Select Keyword File")
        self.setModal(True)
        self.resize(500, 120)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Enter keyword file path or use Browse to select a file:")
        layout.addWidget(instructions)
        
        # File input section
        file_layout = QHBoxLayout()
        
        self.file_input = QLineEdit()
        self.file_input.setText(self.current_file)
        self.file_input.setPlaceholderText("Enter file path or leave empty to clear")
        file_layout.addWidget(self.file_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def browse_file(self):
        """Browse for keyword file"""
        # Start from the last used directory if available
        start_dir = ""
        if self.last_keyword_dir:
            start_dir = self.last_keyword_dir
        elif self.parent_gui and hasattr(self.parent_gui, 'last_keyword_dir') and self.parent_gui.last_keyword_dir:
            start_dir = self.parent_gui.last_keyword_dir
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Keyword File", start_dir, "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self.file_input.setText(filename)
            # Save the directory (not the full path) for next time
            self.last_keyword_dir = str(Path(filename).parent)
            self.save_settings()
    
    def get_keyword_file(self):
        """Get the selected keyword file path"""
        return self.file_input.text().strip()
    
    def load_settings(self):
        """Load settings from parent GUI"""
        try:
            if not self.parent_gui:
                return
                
            # Load keyword directory from parent GUI
            if hasattr(self.parent_gui, 'last_keyword_dir') and self.parent_gui.last_keyword_dir:
                self.last_keyword_dir = self.parent_gui.last_keyword_dir
        except Exception as e:
            print(f"Error loading KeywordFileDialog settings: {e}")
    
    def save_settings(self):
        """Save settings to parent GUI"""
        try:
            if not self.parent_gui:
                return
                
            # Update parent GUI with current directory
            self.parent_gui.last_keyword_dir = self.last_keyword_dir
            # Save to config file
            self.parent_gui.save_settings()
            
        except Exception as e:
            print(f"Error saving KeywordFileDialog settings: {e}")

class ItemDetailsDialog(QDialog):
    """Dialog for selecting SKU from multiple matches and editing quantity"""
    
    def __init__(self, matches: List[Dict], current_quantity: str = "", current_sku: str = "", parent=None, item_data=None, is_deleted=False):
        super().__init__(parent)
        self.matches = matches
        self.current_quantity = current_quantity
        self.current_sku = current_sku
        self.selected_match = None
        self.quantity_override = None
        self.edited_text = None  # Store edited text from search
        self.description_override = None  # Store edited description
        self.item_data = item_data or {}  # Store the original item data
        self.is_deleted = is_deleted  # Track if item is deleted
        self.delete_requested = False  # Track if delete/undelete button was clicked
        
        # Get database mappings from parent GUI
        self.description_mapping = {}
        self.type_mapping = {}
        self.stocking_multiple_mapping = {}
        self.family_mapping = {}
        self.order_status_mapping = {}
        if parent and hasattr(parent, 'description_mapping'):
            self.description_mapping = parent.description_mapping
            self.type_mapping = parent.type_mapping
            self.stocking_multiple_mapping = parent.stocking_multiple_mapping
            self.family_mapping = parent.family_mapping
            self.order_status_mapping = parent.order_status_mapping
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Edit Item Details")
        self.setModal(True)
        self.resize(500, 500)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Item Details")
        layout.addWidget(instructions)
        
        # Quantity section (moved to top)
        quantity_group = QGroupBox("Quantity")
        quantity_layout = QFormLayout(quantity_group)
        
        self.quantity_input = QLineEdit()
        self.quantity_input.setText(self.current_quantity)
        self.quantity_input.setPlaceholderText("Enter quantity (leave empty to keep original)")
        quantity_layout.addRow("Quantity:", self.quantity_input)
        
        layout.addWidget(quantity_group)
        
        # Text editing section
        text_group = QGroupBox("Edit Text and Search")
        text_layout = QVBoxLayout(text_group)
        
        # Text input
        text_input_layout = QHBoxLayout()
        text_input_layout.addWidget(QLabel("Text:"))
        self.text_input = QLineEdit()
        self.text_input.setText(self.item_data.get('processed_text', ''))
        self.text_input.setPlaceholderText("Edit the item text for searching...")
        text_input_layout.addWidget(self.text_input)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_matches)
        text_input_layout.addWidget(self.search_button)
        
        text_layout.addLayout(text_input_layout)
        layout.addWidget(text_group)
        
        # Add custom SKU section
        custom_sku_group = QGroupBox("Add Custom SKU")
        custom_sku_layout = QHBoxLayout(custom_sku_group)
        
        self.custom_sku_input = QLineEdit()
        self.custom_sku_input.setPlaceholderText("Enter SKU to add to list...")
        custom_sku_layout.addWidget(self.custom_sku_input)
        
        add_sku_button = QPushButton("Add SKU")
        add_sku_button.clicked.connect(self.add_custom_sku)
        custom_sku_layout.addWidget(add_sku_button)
        
        layout.addWidget(custom_sku_group)
        
        # Description editing section
        description_group = QGroupBox("Description")
        description_layout = QFormLayout(description_group)
        
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("Description will appear here when an item is selected")
        self.description_input.setEnabled(False)  # Initially disabled
        self.description_input.textChanged.connect(self.on_description_changed)
        description_layout.addRow("Description:", self.description_input)
        
        layout.addWidget(description_group)
        
        # List widget for SKU selection
        self.list_widget = QListWidget()
        
        # Add "No matches" option first
        no_match_item = QListWidgetItem("❌ No matches (clear selection)")
        no_match_item.setData(Qt.ItemDataRole.UserRole, None)
        self.list_widget.addItem(no_match_item)
        
        # Add actual matches
        selected_row = 0  # Default to "No matches"
        for i, match in enumerate(self.matches):
            confidence_symbol = self.get_confidence_symbol(match['confidence'])
            item_text = f"{confidence_symbol} {match['part_number']} - {match['description'][:50]}{'...' if len(match['description']) > 50 else ''}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, match)
            self.list_widget.addItem(item)
            
            # Check if this is the currently selected SKU
            if match['part_number'] == self.current_sku:
                selected_row = i + 1  # +1 because "No matches" is at index 0
        
        # Select the appropriate item
        self.list_widget.setCurrentRow(selected_row)
        
        # Connect double-click to accept selection
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        
        # Connect selection change to update description field
        self.list_widget.currentItemChanged.connect(self.on_selection_changed)
        
        # Initialize description field based on initial selection
        self.on_selection_changed(self.list_widget.currentItem(), None)
            
        layout.addWidget(self.list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Delete/Undelete button
        self.delete_button = QPushButton("Undelete Item" if self.is_deleted else "Delete Item")
        self.delete_button.clicked.connect(self.toggle_delete)
        button_layout.addWidget(self.delete_button)
        
        # Add stretch to push OK/Cancel to the right
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_selection)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Get the widget that currently has focus
            focused_widget = self.focusWidget()
            
            if focused_widget == self.text_input:
                # In text field - perform search
                self.search_matches()
            elif focused_widget == self.custom_sku_input:
                # In custom SKU field - add SKU
                self.add_custom_sku()
            else:
                # In any other field - accept selection (OK)
                self.accept_selection()
        else:
            super().keyPressEvent(event)
        
    def on_selection_changed(self, current_item, previous_item):
        """Handle selection change in the list widget"""
        if current_item:
            match = current_item.data(Qt.ItemDataRole.UserRole)
            if match:
                # Item selected - enable description field and show description
                self.description_input.setEnabled(True)
                self.description_input.setText(match.get('description', ''))
            else:
                # "No matches" selected - disable and clear description field
                self.description_input.setEnabled(False)
                self.description_input.clear()
        else:
            # No item selected - disable and clear description field
            self.description_input.setEnabled(False)
            self.description_input.clear()
        
    def on_description_changed(self):
        """Handle description text changes - update the list box display"""
        current_item = self.list_widget.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole):
            # Update the match data with the new description
            match = current_item.data(Qt.ItemDataRole.UserRole)
            match['description'] = self.description_input.text()
            
            # Update the display text in the list box
            confidence_symbol = self.get_confidence_symbol(match['confidence'])
            item_text = f"{confidence_symbol} {match['part_number']} - {match['description'][:50]}{'...' if len(match['description']) > 50 else ''}"
            current_item.setText(item_text)
        
    def add_custom_sku(self):
        """Add a custom SKU to the matches list"""
        sku_text = self.custom_sku_input.text().strip().upper()
        if not sku_text:
            return
        
        # Check if SKU already exists in the list
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole):
                match = item.data(Qt.ItemDataRole.UserRole)
                if match['part_number'] == sku_text:
                    # SKU already exists, just select it
                    self.list_widget.setCurrentRow(i)
                    self.custom_sku_input.clear()
                    return
        
        # Check if SKU exists in the database
        description = 'Custom SKU (not in database)'
        item_type = ''
        confidence = '1.0'  # High confidence since user explicitly entered it
        
        if sku_text in self.description_mapping:
            # SKU exists in database - use database information
            description = self.description_mapping[sku_text]
            item_type = self.type_mapping.get(sku_text, '')
            confidence = '0.95'  # Slightly lower confidence since it's manually entered but exists in DB
        else:
            # SKU not in database - show warning
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, 
                'SKU Not Found', 
                f'SKU "{sku_text}" was not found in the database.\n\nDo you want to add it as a custom SKU anyway?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Create a new match entry for the custom SKU
        custom_match = {
            'part_number': sku_text,
            'description': description,
            'type': item_type,
            'confidence': confidence
        }
        
        # Add to matches list
        self.matches.append(custom_match)
        
        # Add to list widget
        confidence_symbol = self.get_confidence_symbol(custom_match['confidence'])
        item_text = f"{confidence_symbol} {custom_match['part_number']} - {custom_match['description'][:50]}{'...' if len(custom_match['description']) > 50 else ''}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, custom_match)
        self.list_widget.addItem(item)
        
        # Select the newly added item
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)
        
        # Clear the input field
        self.custom_sku_input.clear()
    
    def search_matches(self):
        """Search for matches using RulesMatcher"""
        try:
            # Get the edited text
            edited_text = self.text_input.text().strip()
            if not edited_text:
                QMessageBox.warning(self, "No Text", "Please enter some text to search for.")
                return
            
            # Show loading state
            self.search_button.setText("Searching...")
            self.search_button.setEnabled(False)
            self.setCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()  # Update UI immediately
            
            # Get quantity
            quantity = self.quantity_input.text().strip() or self.current_quantity
            
            # Create ScannedItem
            scanned_item = ScannedItem(
                quantity=quantity,
                description=edited_text,
                original_text=self.item_data.get('original_text', edited_text),
                matches=[]
            )
            
            # Try to get RulesMatcher from parent
            if not hasattr(self.parent(), 'rules_matcher') or not self.parent().rules_matcher:
                QMessageBox.warning(self, "No Matcher", "RulesMatcher not available. Please load a database first.")
                return
            
            # Search for matches
            matches = self.parent().rules_matcher.match_lumber_item(scanned_item)
            
            if not matches:
                QMessageBox.information(self, "No Matches", f"No matches found for: {edited_text}")
                return
            
            # Show search results dialog
            self.show_search_results(matches)
            
        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Error searching for matches: {str(e)}")
        finally:
            # Restore normal state
            self.search_button.setText("Search")
            self.search_button.setEnabled(True)
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def show_search_results(self, matches):
        """Show search results in a dialog and allow selection"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Search Results")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel(f"Found {len(matches)} matches. Select items to add to the list:")
        layout.addWidget(instructions)
        
        # Results list
        results_list = QListWidget()
        results_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        for match in matches:
            confidence_symbol = self.get_confidence_symbol(str(match.confidence))
            item_text = f"{confidence_symbol} {match.part_number} - {match.description[:60]}{'...' if len(match.description) > 60 else ''}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, match)
            results_list.addItem(item)
        
        # Auto-select the first item
        if results_list.count() > 0:
            results_list.setCurrentRow(0)
        ()
        layout.addWidget(results_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_selected_button = QPushButton("Add Selected")
        add_selected_button.clicked.connect(lambda: self.add_selected_matches(results_list, dialog))
        button_layout.addWidget(add_selected_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def add_selected_matches(self, results_list, dialog):
        """Add selected matches to the main list"""
        selected_items = results_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one item to add.")
            return
        
        added_count = 0
        for item in selected_items:
            match = item.data(Qt.ItemDataRole.UserRole)
            if match:
                # Convert PartMatch to match format
                match_dict = {
                    'part_number': match.part_number,
                    'description': match.description,
                    'type': match.type,
                    'confidence': str(match.confidence)
                }
                
                # Add to matches list
                self.matches.append(match_dict)
                
                # Add to list widget
                confidence_symbol = self.get_confidence_symbol(str(match.confidence))
                item_text = f"{confidence_symbol} {match.part_number} - {match.description[:50]}{'...' if len(match.description) > 50 else ''}"
                list_item = QListWidgetItem(item_text)
                list_item.setData(Qt.ItemDataRole.UserRole, match_dict)
                self.list_widget.addItem(list_item)
                
                added_count += 1
        
        # Select the last added item
        if added_count > 0:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
        
        # Store the edited text for later use
        self.edited_text = self.text_input.text().strip()
        
        dialog.accept()
        QMessageBox.information(self, "Matches Added", f"Added {added_count} matches to the list.")
        
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
    
    def toggle_delete(self):
        """Toggle delete/undelete status and close dialog"""
        self.is_deleted = not self.is_deleted
        self.delete_requested = True
        self.delete_button.setText("Undelete Item" if self.is_deleted else "Delete Item")
        # Close the dialog immediately
        self.accept()
            
    def accept_selection(self):
        """Accept the selected SKU and quantity"""
        current_item = self.list_widget.currentItem()
        if current_item:
            # Get the current match data (which may have been updated by description changes)
            self.selected_match = current_item.data(Qt.ItemDataRole.UserRole)
            
            # Ensure the description in the match data is up to date
            if self.selected_match and self.description_input.isEnabled():
                self.selected_match['description'] = self.description_input.text().strip()
            
            # Handle quantity override
            quantity_text = self.quantity_input.text().strip()
            if quantity_text and quantity_text != self.current_quantity:
                self.quantity_override = quantity_text
            else:
                self.quantity_override = None
            
            # Validate quantity if a match is selected (for new items or when required)
            # Check if this is a new item (empty original quantity) or if quantity is required
            if self.selected_match is not None:
                # A match is selected - quantity is required
                final_quantity = self.quantity_override if self.quantity_override else (quantity_text if quantity_text else self.current_quantity)
                if not final_quantity or not final_quantity.strip():
                    QMessageBox.warning(self, "Quantity Required", 
                                      "Please enter a quantity for this item.")
                    return  # Don't accept the dialog
                # Also check if quantity is zero
                try:
                    # Try to parse as number - check if it's zero
                    qty_value = float(final_quantity.replace(',', '').replace('/', '').split()[0] if final_quantity else '0')
                    if qty_value == 0:
                        QMessageBox.warning(self, "Invalid Quantity", 
                                          "Quantity cannot be zero. Please enter a valid quantity.")
                        return  # Don't accept the dialog
                except (ValueError, AttributeError):
                    # If we can't parse it, that's okay - might be a complex quantity like "2/8"
                    # Just check it's not empty (already done above)
                    pass
            
            # Store the edited text
            self.edited_text = self.text_input.text().strip()
            
            # Store the edited description - since on_description_changed updates the match data,
            # we just need to check if the description field was enabled (meaning an item was selected)
            if self.description_input.isEnabled():
                self.description_override = self.description_input.text().strip()
            else:
                self.description_override = None
                
            self.accept()
        else:
            self.reject()

class DocumentProcessor(QObject):
    """Worker thread for processing documents"""
    finished = pyqtSignal(object, object)  # Signal with databases and scanned_items
    error = pyqtSignal(str)
    notify = pyqtSignal(str)  # Signal for status updates
    
    def __init__(self, pdf_file, api_key, database_files, output_dir, processing_dialog, keyword_file=None):
        super().__init__()
        self.pdf_file = pdf_file
        self.api_key = api_key
        self.database_files = database_files
        self.output_dir = output_dir
        self.processing_dialog = processing_dialog
        self.keyword_file = keyword_file
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
            
            # Set keyword file environment variable if provided
            if self.keyword_file and os.path.exists(self.keyword_file):
                os.environ["MATCHER_KEYWORDS"] = self.keyword_file
            
            # Run the matcher with notify and error functions
            result = run_matcher(
                document=self.pdf_file,
                api_key=self.api_key,
                database_names=self.database_files,
                training_data=None,
                use_ai_matching=False,
                output_dir=output_dir_path,
                output_file_name="matches.csv",
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
    
    def __init__(self, pdf_file, api_key, database_files, output_dir, parent=None, keyword_file=None):
        super().__init__(parent)
        self.pdf_file = pdf_file
        self.api_key = api_key
        self.database_files = database_files
        self.output_dir = output_dir
        self.keyword_file = keyword_file
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
            self.pdf_file, self.api_key, self.database_files, self.output_dir, self, self.keyword_file
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
            pdf_file, api_key, self.database_files, str(output_dir), self.parent(), self.parent_gui.keyword_file
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
        self.sku_overrides = {}  # Store manual SKU selections
        self.quantity_overrides = {}  # Store manual quantity overrides
        self.text_overrides = {}  # Store manual text overrides
        self.description_overrides = {}  # Store manual description overrides
        self.deleted_items = set()  # Store deleted items by item_number
        
        # Config file path
        self.config_file = Path.home() / ".jlc.ini"
        
        # Initialize settings attributes
        self.last_api_key = ""
        self.last_pdf_dir = ""
        self.last_db_dir = ""
        self.last_database_files = []
        self.keyword_file = ""
        self.last_keyword_dir = ""
        
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
            config['Options'] = {}
            
            # Save ProcessDocument settings (will be populated by ProcessDocumentDialog)
            if hasattr(self, 'last_api_key') and self.last_api_key:
                config['ProcessDocument']['api_key'] = self.last_api_key
            if hasattr(self, 'last_pdf_dir') and self.last_pdf_dir:
                config['ProcessDocument']['pdf_dir'] = self.last_pdf_dir
            if hasattr(self, 'last_db_dir') and self.last_db_dir:
                config['ProcessDocument']['db_dir'] = self.last_db_dir
            if hasattr(self, 'last_database_files') and self.last_database_files:
                config['ProcessDocument']['database_files'] = '|'.join(self.last_database_files)
            
            # Save Options settings
            if hasattr(self, 'keyword_file') and self.keyword_file:
                config['Options']['keyword_file'] = self.keyword_file
            if hasattr(self, 'last_keyword_dir') and self.last_keyword_dir:
                config['Options']['keyword_dir'] = self.last_keyword_dir
            
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
            
            # Load Options settings
            if 'Options' in config:
                self.keyword_file = config.get('Options', 'keyword_file', fallback='')
                self.last_keyword_dir = config.get('Options', 'keyword_dir', fallback='')
            
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

        self.no_match_sku = os.getenv("NO_MATCH_SKU") if os.getenv("NO_MATCH_SKU") is not None else "20"

    def closeEvent(self, event):
        """Handle application close event"""
        if self.ask_save_changes():
            self.save_settings()
            event.accept()
        else:
            event.ignore()  # Cancel the close operation
        
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
        
        # New Item action
        new_item_action = QAction('&New Item', self)
        new_item_action.setShortcut('Ctrl+N')
        new_item_action.setStatusTip('Add a new item to the list')
        new_item_action.triggered.connect(self.add_new_item)
        file_menu.addAction(new_item_action)
        
        file_menu.addSeparator()
        
        # Load CSV action
        load_csv_action = QAction('&Open Results CSV...', self)
        load_csv_action.setShortcut('Ctrl+O')
        load_csv_action.setStatusTip('Load matching results CSV file')
        load_csv_action.triggered.connect(self.load_csv_file)
        file_menu.addAction(load_csv_action)
        
        # Save action
        save_action = QAction('&Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save changes to current CSV file')
        save_action.triggered.connect(self.save_csv_file)
        save_action.setEnabled(False)
        self.save_action = save_action  # Store reference for enabling/disabling
        file_menu.addAction(save_action)
        
        # Export action
        export_action = QAction('&Save as...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.setStatusTip('Save results to new CSV')
        export_action.triggered.connect(self.export_to_csv)
        export_action.setEnabled(False)
        self.export_action = export_action  # Store reference for enabling/disabling
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()

        # Export to POS System action
        export_pos_action = QAction('Export to &POS System...', self)
        export_pos_action.setShortcut('Ctrl+R')
        export_pos_action.setStatusTip('Export to POS System format (SKU, Qty, Linear Feet)')
        export_pos_action.triggered.connect(self.export_to_pos_system)
        export_pos_action.setEnabled(False)
        self.export_pos_action = export_pos_action  # Store reference for enabling/disabling
        file_menu.addAction(export_pos_action)
        
        file_menu.addSeparator()

        # Load Database action
        load_db_action = QAction('Load &Database...', self)
        load_db_action.setShortcut('Ctrl+D')
        load_db_action.setStatusTip('Load SKU database file')
        load_db_action.triggered.connect(self.load_database_file)
        file_menu.addAction(load_db_action)
        
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
        self.show_unique_matches_action = QAction('Show Items with &Unique Matches', self)
        self.show_unique_matches_action.setCheckable(True)
        self.show_unique_matches_action.setChecked(True)
        self.show_unique_matches_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_unique_matches_action)
        
        self.show_multiple_matches_action = QAction('Show Items with &Multiple Matches', self)
        self.show_multiple_matches_action.setCheckable(True)
        self.show_multiple_matches_action.setChecked(True)
        self.show_multiple_matches_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_multiple_matches_action)
        
        self.show_no_matches_action = QAction('Show Items with &No Matches', self)
        self.show_no_matches_action.setCheckable(True)
        self.show_no_matches_action.setChecked(True)
        self.show_no_matches_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_no_matches_action)
        
        view_menu.addSeparator()  # Add separator for Stock filters
        
        self.show_stock_items_action = QAction('Show &Stock Items', self)
        self.show_stock_items_action.setCheckable(True)
        self.show_stock_items_action.setChecked(True)
        self.show_stock_items_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_stock_items_action)
        
        self.show_so_items_action = QAction('Show &S/O Items', self)
        self.show_so_items_action.setCheckable(True)
        self.show_so_items_action.setChecked(True)
        self.show_so_items_action.triggered.connect(self.apply_filters)
        view_menu.addAction(self.show_so_items_action)
        
        # Options menu
        options_menu = menubar.addMenu('&Options')
        
        # Keyword file action
        keyword_file_action = QAction('&Keyword File...', self)
        keyword_file_action.setStatusTip('Select keyword file for matching')
        keyword_file_action.triggered.connect(self.select_keyword_file)
        options_menu.addAction(keyword_file_action)
        
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
        
        # New Item button (moved to left side)
        new_item_action = QAction('➕ New Item', self)
        new_item_action.setStatusTip('Add a new item to the list')
        new_item_action.triggered.connect(self.add_new_item)
        toolbar.addAction(new_item_action)
        
        toolbar.addSeparator()
        
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
            "Item #", "Quantity", 
            "SKU", "Description", "Type", "Stk/SO", "Confidence",
            "Text", "Original"
        ]
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        
        # Set up custom delegate for deleted items (after columns are defined)
        self.deleted_delegate = DeletedItemDelegate(self.table, self.deleted_items)
        self.deleted_delegate.set_row_item_data(self.row_item_data)
        # Apply delegate to all columns
        for col in range(len(self.columns)):
            self.table.setItemDelegateForColumn(col, self.deleted_delegate)

        self.column_numbers = {
            "Item #": 0, "Quantity": 1, 
            "SKU": 2, "Description": 3, "Type": 4, "Stk/SO": 5, "Confidence": 6,
            "Text": 7, "Original": 8
        }
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.column_numbers["Item #"], QHeaderView.ResizeMode.Fixed)  # Item #
        header.setSectionResizeMode(self.column_numbers["Quantity"], QHeaderView.ResizeMode.Fixed)  # Quantity
        header.setSectionResizeMode(self.column_numbers["SKU"], QHeaderView.ResizeMode.Fixed)  # SKU
        header.setSectionResizeMode(self.column_numbers["Description"], QHeaderView.ResizeMode.Stretch)  # Description
        header.setSectionResizeMode(self.column_numbers["Type"], QHeaderView.ResizeMode.Fixed)  # Type
        header.setSectionResizeMode(self.column_numbers["Stk/SO"], QHeaderView.ResizeMode.Fixed)  # Stk/SO
        header.setSectionResizeMode(self.column_numbers["Confidence"], QHeaderView.ResizeMode.Fixed)  # Confidence
        header.setSectionResizeMode(self.column_numbers["Text"], QHeaderView.ResizeMode.Stretch)  # Text
        header.setSectionResizeMode(self.column_numbers["Original"], QHeaderView.ResizeMode.Stretch)  # Original
        
        self.table.setColumnWidth(self.column_numbers["Item #"], 60)   # Item #
        self.table.setColumnWidth(self.column_numbers["Quantity"], 80)   # Quantity
        self.table.setColumnWidth(self.column_numbers["SKU"], 120)  # SKU
        self.table.setColumnWidth(self.column_numbers["Type"], 100)  # Type
        self.table.setColumnWidth(self.column_numbers["Stk/SO"], 60)  # Stk/SO
        self.table.setColumnWidth(self.column_numbers["Confidence"], 100)  # Confidence
        
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
        if not self.ask_save_changes():
            return  # User cancelled
        
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
        if not self.ask_save_changes():
            return  # User cancelled
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load SKU Database", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            self.database_file = filename
            self.load_data()
    
    def write_csv_file(self, csv_filename):
        # Write the updated data back to the CSV file
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header using original CSV columns
            if hasattr(self, 'original_csv_columns') and self.original_csv_columns:
                writer.writerow(self.original_csv_columns)
            else:
                # Fallback to basic columns
                writer.writerow(['Item_Number', 'Quantity', 'Description', 'Original_Text', 'Part_Number', 'Confidence'])
            
            # Write data rows - need to work with raw_data but apply overrides from grouped_data
            skip_item = 0       # if we have an override, we use that and skip any other matches
            for raw_row in self.raw_data:
                # Find the corresponding grouped item to check for overrides
                item_number = raw_row.get('item_number', '')
                
                # Skip deleted items
                if item_number in self.deleted_items:
                    continue
                
                if item_number == skip_item:
                    continue
                
                # Find the display row index using row_item_data mapping
                # Also check grouped_data for newly added items that might not be in row_item_data yet
                display_row_idx = None
                for row_idx, item in self.row_item_data.items():
                    if item.get('item_number') == item_number:
                        display_row_idx = row_idx
                        break
                
                # If not found in row_item_data, check if it's a newly added item in grouped_data
                # and find its index in filtered_data (which is what row_item_data is based on)
                if display_row_idx is None:
                    # Try to find it in filtered_data to get the correct index
                    for idx, filtered_item in enumerate(self.filtered_data):
                        if filtered_item.get('item_number') == item_number:
                            display_row_idx = idx
                            break
                
                # Apply overrides if this item has them
                updated_row = raw_row.copy()
                if display_row_idx is not None:
                    # Apply quantity override
                    if display_row_idx in self.quantity_overrides:
                        updated_row['Quantity'] = self.quantity_overrides[display_row_idx]
                    
                    # Apply text override
                    if display_row_idx in self.text_overrides:
                        updated_row['Description'] = self.text_overrides[display_row_idx]
                    
                    # Apply description override
                    if display_row_idx in self.description_overrides:
                        updated_row['Database_Description'] = self.description_overrides[display_row_idx]
                    
                    # Apply SKU override
                    if display_row_idx in self.sku_overrides:
                        skip_item = item_number         # only write the overridden item and no other matches
                        override = self.sku_overrides[display_row_idx]
                        if override is None:
                            # "No matches" was selected
                            updated_row['Part_Number'] = ''
                            updated_row['Confidence'] = ''
                        else:
                            # Specific match was selected
                            updated_row['Part_Number'] = override['part_number']
                            updated_row['Confidence'] = override['confidence']
                            updated_row['Database_Description'] = self._insert_db_description(override['description'],
                                                                                  updated_row['Database_Description'])
                
                # Write the row
                # For newly added items, ensure all fields are properly set
                if hasattr(self, 'original_csv_columns') and self.original_csv_columns:
                    row_values = [updated_row.get(col, '') for col in self.original_csv_columns]
                else:
                    # Fallback to basic values
                    row_values = [
                        updated_row.get('Item_Number', ''),
                        updated_row.get('Quantity', ''),
                        updated_row.get('Description', updated_row.get('Processed_Text', '')),
                        updated_row.get('Original_Text', ''),
                        updated_row.get('Part_Number', ''),
                        updated_row.get('Confidence', ''),
                    ]
                writer.writerow(row_values)

    def save_csv_file(self):
        """Save changes back to the current CSV file"""
        if not self.csv_file:
            QMessageBox.warning(self, "No File", "No CSV file is currently loaded.")
            return
        
        if not self.raw_data:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        
        try:
            # Create a backup of the original file
            backup_file = self.csv_file + ".backup"
            import shutil
            shutil.copy2(self.csv_file, backup_file)
            
            self.write_csv_file(self.csv_file)
            
            QMessageBox.information(self, "Save Complete", 
                                  f"Changes saved to {Path(self.csv_file).name}\n"
                                  f"Backup created: {Path(backup_file).name}")
            
            # Clear overrides and reload the CSV to sync state
            self.sku_overrides.clear()
            self.quantity_overrides.clear()
            self.text_overrides.clear()
            self.description_overrides.clear()
            self.load_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")
    
    def has_unsaved_changes(self):
        """Check if there are any unsaved changes (manual overrides)"""
        return (len(self.sku_overrides) > 0 or len(self.quantity_overrides) > 0 or 
                len(self.text_overrides) > 0 or len(self.description_overrides) > 0)
    
    def ask_save_changes(self):
        """Ask user if they want to save changes before proceeding"""
        if not self.has_unsaved_changes():
            return True  # No changes to save
        
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save
        )
        
        if reply == QMessageBox.StandardButton.Save:
            self.save_csv_file()
            return True
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:  # Cancel
            return False
    
    def apply_overrides_to_row(self, row):
        """Apply manual overrides to a single row - DEPRECATED, use save_csv_file instead"""
        # This method is no longer used - overrides are applied directly in save_csv_file
        return row
    
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
            self.sku_overrides.clear()
            self.quantity_overrides.clear()
            self.text_overrides.clear()
            self.description_overrides.clear()
            self.row_item_data.clear()
            self.sku_comboboxes.clear()
            
            # Load database
            (self.description_mapping, self.type_mapping, 
             self.stocking_multiple_mapping, self.family_mapping,
             self.order_status_mapping) = self.load_database(self.database_file)
            
            # Initialize RulesMatcher
            self.init_rules_matcher()
            
            # Load and process CSV
            self.raw_data = self.process_csv_file(self.csv_file)
            self.group_data()
            self.apply_filters()
            self.update_display()
            self.update_window_title()  # Update window title with CSV filename
            
            self.export_action.setEnabled(True)
            self.export_pos_action.setEnabled(True)
            self.save_action.setEnabled(True)
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
        part_to_family = {}
        part_to_order = {}
        term_to_type = {}
        
        with open(database_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                part_number = row.get('Item Number', '').strip()
                description = row.get('Item Description', '').strip()
                stocking_multiple = row.get('Stocking Multiple', '').strip()
                family = row.get("Material Type", "").strip()
                order = row.get("Stock or SO", "").strip()
                
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
                    part_to_family[part_number] = family
                    part_to_order[part_number] = order
                    
        return part_to_description, part_to_type, part_to_stocking_multiple, part_to_family, part_to_order
    
    def init_rules_matcher(self):
        """Initialize RulesMatcher for search functionality"""
        try:
            # Import required modules
            from pdf2parts import load_database
            from match import RulesMatcher
            
            # Load database for RulesMatcher
            db_name = Path(self.database_file).stem
            database = load_database(self.database_file, db_name, quiet=True)
            
            # Create matcherdbs dict
            matcherdbs = {self.database_file: database}
            
            # Initialize RulesMatcher
            self.rules_matcher = RulesMatcher(matcherdbs, False)
            
        except Exception as e:
            print(f"Warning: Could not initialize RulesMatcher: {e}")
            self.rules_matcher = None
        
    def _extract_db_description(self, db_desc: str) -> str:
        divider = db_desc.find('|')
        if divider >= 0:
            item_description = db_desc[:divider]
            return item_description.strip()
        return db_desc

    def _insert_db_description(self, item_description: str, db_desc: str) -> str:
        divider = db_desc.find('|')
        if divider >= 0:
            db_desc = item_description.strip() + ' ' + db_desc[divider:]
        else:
            # must be empty
            db_desc = item_description.strip() + ' | No other information available'

        return db_desc

    def process_csv_file(self, csv_file: str) -> List[Dict]:
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
                database_description = row.get('Database_Description', '').strip()
                confidence = row.get('Confidence', '').strip()
                
                # extract the description and type from the Database_Description field
                #
                item_description = self._extract_db_description(row.get("Database_Description"))
                item_type = ""
                if part_number and part_number in self.type_mapping:
                    item_type = self.type_mapping[part_number]

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
            show_unique_matches = self.show_unique_matches_action.isChecked()
            show_multiple_matches = self.show_multiple_matches_action.isChecked()
            show_no_matches = self.show_no_matches_action.isChecked()
            show_stock_items = self.show_stock_items_action.isChecked()
            show_so_items = self.show_so_items_action.isChecked()
            
            self.filtered_data = []
            
            for i, item in enumerate(self.grouped_data):
                
                # Check search filter
                if search_text:
                    searchable_text = f"{item['item_number']} {item['processed_text']} {item['original_text']}".lower()
                    if search_text not in searchable_text:
                        continue
                
                # Check match status filter
                has_matches = len(item['matches']) > 0
                if has_matches:
                    # Item has matches - check if it's unique or multiple
                    if len(item['matches']) == 1:
                        # Unique match
                        if not show_unique_matches:
                            continue
                    else:
                        # Multiple matches
                        if not show_multiple_matches:
                            continue
                else:
                    # No matches
                    if not show_no_matches:
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
                
                # Check Stock/SO filter
                # Get the SKU from the first match (or override if available)
                selected_sku = None
                if has_matches and item['matches']:
                    # Use first match's SKU to determine Stock/SO status
                    selected_sku = item['matches'][0].get('part_number')
                
                # Check the Stock/SO status if we have a SKU
                if selected_sku and selected_sku in self.order_status_mapping:
                    stk_so_value = self.order_status_mapping[selected_sku]
                    if stk_so_value == "Stock" and not show_stock_items:
                        continue
                    if stk_so_value == "S/O" and not show_so_items:
                        continue
                # If no SKU or SKU not in mapping, include the item (don't filter it out)
                
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
        self.show_unique_matches_action.setChecked(True)
        self.show_multiple_matches_action.setChecked(True)
        self.show_no_matches_action.setChecked(True)
        self.show_stock_items_action.setChecked(True)
        self.show_so_items_action.setChecked(True)
        self.apply_filters()
        
    def select_keyword_file(self):
        """Open keyword file selection dialog with text box and browse button"""
        dialog = KeywordFileDialog(self.keyword_file, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.keyword_file = dialog.get_keyword_file()
            self.save_settings()
            if self.keyword_file:
                QMessageBox.information(self, "Keyword File Updated", 
                                      f"Keyword file set to:\n{self.keyword_file}")
            else:
                QMessageBox.information(self, "Keyword File Cleared", 
                                      "Keyword file has been cleared.")
        
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
                    self.table.setItem(row_idx, self.column_numbers["Item #"], QTableWidgetItem(str(item['item_number'])))
                    
                    # Handle quantity display with override support
                    quantity_text = str(item['quantity'])
                    if row_idx in self.quantity_overrides:
                        quantity_text = f"🔧 {self.quantity_overrides[row_idx]}"
                    
                    quantity_item = QTableWidgetItem(quantity_text)
                    if row_idx in self.quantity_overrides:
                        # Highlight overridden quantities
                        font = quantity_item.font()
                        font.setBold(True)
                        quantity_item.setFont(font)
                        quantity_item.setBackground(QColor(255, 255, 200))  # Light yellow background
                        quantity_item.setToolTip(f"Original: {item['quantity']}\nOverride: {self.quantity_overrides[row_idx]}")
                    else:
                        quantity_item.setToolTip("Double-click to edit quantity")
                    
                    self.table.setItem(row_idx, self.column_numbers["Quantity"], quantity_item)
                    
                    # Handle text display with override support
                    text_to_display = item['processed_text']
                    if row_idx in self.text_overrides:
                        text_to_display = self.text_overrides[row_idx]
                    
                    text_item = QTableWidgetItem(text_to_display)
                    if row_idx in self.text_overrides:
                        # Highlight overridden text
                        font = text_item.font()
                        font.setBold(True)
                        text_item.setFont(font)
                        text_item.setBackground(QColor(255, 255, 200))  # Light yellow background
                        text_item.setToolTip(f"Original: {item['processed_text']}\nOverride: {self.text_overrides[row_idx]}")
                    
                    self.table.setItem(row_idx, self.column_numbers["Text"], text_item)
                    self.table.setItem(row_idx, self.column_numbers["Original"], QTableWidgetItem(item['original_text']))
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
            
            # Update the deleted delegate with current data
            self.update_deleted_delegate()
            
        except Exception as e:
            print(f"Error in update_display: {e}")
            self.status_bar.showMessage("Error updating display")
        
    def on_cell_clicked(self, row: int, column: int):
        """Handle single click on table cell"""
        if row in self.row_item_data:
            item = self.row_item_data[row]
            # Always open dialog for any item (multiple matches, single match, or no matches)
            self.open_sku_dialog(row, item)
    
    def on_cell_double_clicked(self, row: int, column: int):
        """Handle double-click on table cell"""
        if row in self.row_item_data:
            item = self.row_item_data[row]
            # Always open dialog for any item (multiple matches, single match, or no matches)
            self.open_sku_dialog(row, item)
    
    def open_sku_dialog(self, row: int, item: dict):
        """Open SKU selection dialog for an item"""
        # Get current quantity (original or override)
        current_quantity = self.quantity_overrides.get(row, item['quantity'])
        
        # Get current SKU (from manual override or first match)
        current_sku = ""
        if row in self.sku_overrides and self.sku_overrides[row] is not None:
            current_sku = self.sku_overrides[row]['part_number']
        elif item['matches']:
            current_sku = item['matches'][0]['part_number']
        
        # Check if item is deleted
        is_deleted = item.get('item_number') in self.deleted_items
        
        dialog = ItemDetailsDialog(item['matches'], current_quantity, current_sku, self, item, is_deleted)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Check if text was edited - if so, this should be treated as a manual override
            text_was_edited = (dialog.edited_text is not None and 
                             dialog.edited_text != item['processed_text'])
            
            if dialog.selected_match is None:
                # "No matches" selected - store as override
                self.sku_overrides[row] = None
            else:
                # Match selected - store as override
                self.sku_overrides[row] = dialog.selected_match
            
            # Handle quantity override
            if dialog.quantity_override is not None:
                self.quantity_overrides[row] = dialog.quantity_override
            elif row in self.quantity_overrides:
                # User cleared the quantity - remove override
                del self.quantity_overrides[row]
            
            # Handle text override
            if text_was_edited:
                self.text_overrides[row] = dialog.edited_text
                # If text was edited and a match was selected, ensure it's in sku_overrides
                if dialog.selected_match is not None and row not in self.sku_overrides:
                    self.sku_overrides[row] = dialog.selected_match
            elif row in self.text_overrides:
                # User didn't change text - remove override
                del self.text_overrides[row]
            
            # Handle description override
            if dialog.description_override is not None:
                self.description_overrides[row] = dialog.description_override
                # If description was edited and a match was selected, ensure it's in sku_overrides
                if dialog.selected_match is not None and row not in self.sku_overrides:
                    self.sku_overrides[row] = dialog.selected_match
            elif row in self.description_overrides:
                # User didn't change description - remove override
                del self.description_overrides[row]
            
            # Handle delete/undelete action
            if dialog.delete_requested:
                item_number = item.get('item_number')
                if item_number:
                    if dialog.is_deleted:
                        self.deleted_items.add(item_number)
                    else:
                        self.deleted_items.discard(item_number)
                # Update delegate with new deleted items
                self.update_deleted_delegate()
            
            self.update_display_for_row(row, item)
    
    def add_new_item(self):
        """Add a new item to the list"""
        if not self.grouped_data:
            # If no data loaded, start with item number 1
            new_item_number = "1"
        else:
            # Find the maximum item number
            max_item_number = 0
            for item in self.grouped_data:
                try:
                    item_num = int(item.get('item_number', '0'))
                    max_item_number = max(max_item_number, item_num)
                except (ValueError, TypeError):
                    continue
            new_item_number = str(max_item_number + 1)
        
        # Create a new empty item
        new_item = {
            'item_number': new_item_number,
            'quantity': '',
            'processed_text': '',
            'original_text': '',
            'matches': []
        }
        
        # Open the item details dialog
        dialog = ItemDetailsDialog([], '', '', self, new_item, False)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Check if a match was selected
            if dialog.selected_match is not None:
                # Quantity validation is now done in the dialog's accept_selection method
                # Get the quantity from the dialog
                quantity = dialog.quantity_override if dialog.quantity_override else ''
                
                # Update the item with dialog data
                new_item['quantity'] = quantity
                new_item['processed_text'] = dialog.edited_text if dialog.edited_text else ''
                new_item['original_text'] = dialog.edited_text if dialog.edited_text else ''
                new_item['matches'] = [dialog.selected_match]
                
                # Add to raw_data (create a row for CSV export)
                # Format Database_Description properly
                description = dialog.selected_match.get('description', '')
                item_type = dialog.selected_match.get('type', '')
                # Use the same format as in process_csv_file
                db_description = description
                if item_type:
                    db_description = f"{description} | {item_type}"
                else:
                    db_description = f"{description} | No other information available"
                
                new_raw_row = {
                    'Item_Number': new_item_number,
                    'Quantity': new_item['quantity'],
                    'Description': new_item['processed_text'],
                    'Original_Text': new_item['original_text'],
                    'Part_Number': dialog.selected_match['part_number'],
                    'Database_Description': db_description,
                    'Confidence': dialog.selected_match.get('confidence', ''),
                    'item_number': new_item_number,
                    'quantity': new_item['quantity'],
                    'processed_text': new_item['processed_text'],
                    'original_text': new_item['original_text'],
                    'part_number': dialog.selected_match['part_number'],
                    'item_description': description,
                    'item_type': item_type,
                    'confidence': dialog.selected_match.get('confidence', ''),
                }
                
                # Preserve original CSV columns if they exist
                if hasattr(self, 'original_csv_columns') and self.original_csv_columns:
                    for col in self.original_csv_columns:
                        if col not in new_raw_row:
                            new_raw_row[col] = ''
                
                self.raw_data.append(new_raw_row)
                
                # Add to grouped_data
                self.grouped_data.append(new_item)
                
                # Refresh the display
                self.apply_filters()
                self.update_display()
                
                # Scroll to the new item (it should be at the end)
                if self.filtered_data:
                    last_row = len(self.filtered_data) - 1
                    self.table.scrollToItem(self.table.item(last_row, 0))
                    self.table.selectRow(last_row)
            else:
                # User clicked OK but didn't select a match - don't add the item
                QMessageBox.information(self, "No Selection", 
                                      "Please select a SKU match to add the item.")
    
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
    
    def update_deleted_delegate(self):
        """Update the deleted delegate with current data"""
        if hasattr(self, 'deleted_delegate'):
            self.deleted_delegate.set_deleted_items(self.deleted_items)
            self.deleted_delegate.set_row_item_data(self.row_item_data)
    
    def update_display_for_row(self, row: int, item: dict):
        """Update display for a single row"""
        if row not in self.row_item_data:
            return
        
        # Check if item is deleted
        is_deleted = item.get('item_number') in self.deleted_items
        
        # Update quantity column first
        quantity_text = str(item['quantity'])
        if row in self.quantity_overrides:
            quantity_text = f"🔧 {self.quantity_overrides[row]}"
        
        quantity_item = QTableWidgetItem(quantity_text)
        if row in self.quantity_overrides:
            # Highlight overridden quantities
            font = quantity_item.font()
            font.setBold(True)
            quantity_item.setFont(font)
            quantity_item.setBackground(QColor(255, 255, 200))  # Light yellow background
            quantity_item.setToolTip(f"Original: {item['quantity']}\nOverride: {self.quantity_overrides[row]}")
        else:
            quantity_item.setToolTip("Double-click to edit quantity")
        
        self.table.setItem(row, self.column_numbers["Quantity"], quantity_item)
        
        # Update text column with override support
        text_to_display = item['processed_text']
        if row in self.text_overrides:
            text_to_display = self.text_overrides[row]
        
        text_item = QTableWidgetItem(text_to_display)
        if row in self.text_overrides:
            # Highlight overridden text
            font = text_item.font()
            font.setBold(True)
            text_item.setFont(font)
            text_item.setBackground(QColor(255, 255, 200))  # Light yellow background
            text_item.setToolTip(f"Original: {item['processed_text']}\nOverride: {self.text_overrides[row]}")
        
        self.table.setItem(row, self.column_numbers["Text"], text_item)
            
        # Check if there's a manual override
        if row in self.sku_overrides:
            override = self.sku_overrides[row]
            if override is None:
                # "No matches" was selected - show bold text and clear fields
                sku_item = QTableWidgetItem("No matches")
                font = sku_item.font()
                font.setBold(True)
                sku_item.setFont(font)
                self.table.setItem(row, self.column_numbers["SKU"], sku_item)
                # Clear description and other fields using update_item_details
                self.update_item_details(row, None)
            else:
                # Show manual override with bold text (user made a choice)
                sku_item = QTableWidgetItem(f"🔧 {override['part_number']}")
                font = sku_item.font()
                font.setBold(True)
                sku_item.setFont(font)
                self.table.setItem(row, self.column_numbers["SKU"], sku_item)
                self.update_item_details(row, override)
            
            # Highlight the entire row for manual overrides AFTER updating all items
            self.highlight_row(row)
        elif len(item['matches']) > 1:
            # Multiple matches - show clickable indicator with bold text (needs user action)
            sku_item = QTableWidgetItem(f"📋 {len(item['matches'])} matches (click to select)")
            font = sku_item.font()
            font.setBold(True)
            sku_item.setFont(font)
            self.table.setItem(row, self.column_numbers["SKU"], sku_item)
            # Set initial values for the first match
            if item['matches']:
                self.update_item_details(row, item['matches'][0])
        elif len(item['matches']) == 1:
            # Single match - show normal text (no action needed)
            match = item['matches'][0]
            sku_item = QTableWidgetItem(match['part_number'])
            self.table.setItem(row, self.column_numbers["SKU"], sku_item)
            self.update_item_details(row, match)
        else:
            # No matches - show normal text (no action needed)
            sku_item = QTableWidgetItem("No matches")
            self.table.setItem(row, self.column_numbers["SKU"], sku_item)
            self.table.setItem(row, self.column_numbers["Description"], QTableWidgetItem(""))
            self.table.setItem(row, self.column_numbers["Type"], QTableWidgetItem(""))
            self.table.setItem(row, self.column_numbers["Confidence"], QTableWidgetItem(""))
            self.update_stk_so_column(row, None)  # Clear Stk/SO column
        
        # Delegate will handle drawing the line for deleted items
        # Just update the delegate reference
        if hasattr(self, 'deleted_delegate'):
            self.deleted_delegate.set_row_item_data(self.row_item_data)
    
    def on_sku_selected(self, row_idx: int, match: Dict):
        """Handle SKU selection from combobox"""
        try:
            self.update_item_details(row_idx, match)
        except Exception as e:
            pass
    
    def update_stk_so_column(self, row_idx: int, sku: str = None):
        """Update the Stk/SO column based on SKU"""
        try:
            if sku and sku in self.order_status_mapping:
                stk_so_value = self.order_status_mapping[sku]
                item = QTableWidgetItem(str(stk_so_value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, self.column_numbers["Stk/SO"], item)
            else:
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, self.column_numbers["Stk/SO"], item)
        except Exception as e:
            # Set empty if there's an error
            try:
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, self.column_numbers["Stk/SO"], item)
            except Exception:
                pass
        
    def update_item_details(self, row_idx: int, match: Dict):
        """Update the description, type, and confidence columns"""
        try:
            if match is None:
                # "No matches" selected - clear all fields
                self.table.setItem(row_idx, self.column_numbers["Description"], QTableWidgetItem(""))
                self.table.setItem(row_idx, self.column_numbers["Type"], QTableWidgetItem(""))
                self.table.setItem(row_idx, self.column_numbers["Confidence"], QTableWidgetItem(""))
                self.update_stk_so_column(row_idx, None)  # Clear Stk/SO column
            else:
                # Normal match - update fields
                description = match.get('description', '')
                item_type = match.get('type', '')
                confidence = match.get('confidence', '')
                sku = match.get('part_number', '')
                
                self.table.setItem(row_idx, self.column_numbers["Description"], QTableWidgetItem(str(description)))
                self.table.setItem(row_idx, self.column_numbers["Type"], QTableWidgetItem(str(item_type)))
                
                confidence_symbol = self.get_confidence_symbol(confidence)
                # Format confidence to 2 decimal places
                try:
                    confidence_float = float(confidence)
                    confidence_text = f"{confidence_symbol} {confidence_float:.2f}"
                except (ValueError, TypeError):
                    confidence_text = f"{confidence_symbol} {confidence}"
                
                self.table.setItem(row_idx, self.column_numbers["Confidence"], QTableWidgetItem(confidence_text))
                self.update_stk_so_column(row_idx, sku)  # Update Stk/SO column
        except Exception as e:
            # Set safe defaults
            try:
                self.table.setItem(row_idx, self.column_numbers["Description"], QTableWidgetItem(""))
                self.table.setItem(row_idx, self.column_numbers["Type"], QTableWidgetItem(""))
                self.table.setItem(row_idx, self.column_numbers["Confidence"], QTableWidgetItem("⚠"))
                self.update_stk_so_column(row_idx, None)  # Clear Stk/SO column
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
                self.write_csv_file(filename)
                QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
    
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
                    # writer.writerow(['SKU', 'Description', 'Qty', 'Linear Feet'])
                    
                    for family in ["L", "H", ""]:
                        for i, item in enumerate(self.filtered_data):
                            # Skip deleted items
                            if item.get('item_number') in self.deleted_items:
                                continue
                            
                            # Check if there's a manual override for this row
                            if i in self.sku_overrides:
                                override = self.sku_overrides[i]
                                if override is None:
                                    # "No matches" was selected - skip this item
                                    if self.no_match_sku:
                                        sku = self.no_match_sku
                                        description = "No match found"
                                    else:
                                        continue
                                else:
                                    # Use the manual override
                                    sku = override['part_number']
                                    description = override['description']
                            elif len(item['matches']) > 0:
                                # No manual override - use the first match
                                match = item['matches'][0]
                                sku = match['part_number']
                                description = match['description']
                            elif self.no_match_sku:
                                sku = self.no_match_sku
                                description = "No match found"
                            else:
                                # Skip items with no matches
                                continue

                            if self.family_mapping.get(sku, "") != family:
                                continue

                            # Use overridden quantity if available
                            quantity = item.get('quantity', '')
                            if i in self.quantity_overrides:
                                quantity = self.quantity_overrides[i]
                            
                            stocking_multiple = self.stocking_multiple_mapping.get(sku, '')
                            row = [sku, description, quantity, ""]
                            if stocking_multiple.lower() == "lf" and (',' in quantity or '/' in quantity):
                                row = row[:2]
                                # should be a comma-separated list with a slash separating the qty from the length
                                for q in quantity.split(','):
                                    slash = q.find('/')
                                    if slash > 0:
                                        row.append(q[:slash].strip())
                                        row.append(q[slash+1:].strip())
                            writer.writerow(row)
                            
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
