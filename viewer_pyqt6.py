#!/usr/bin/env python3
"""
PyQt6 GUI Viewer for Lumber List Matching Results

A modern GUI application with interactive SKU selection for multiple matches.
Features a table view with embedded listboxes for items with multiple SKU matches.
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QCheckBox, QHeaderView, QMessageBox, 
    QFileDialog, QStatusBar, QSplitter, QTextEdit, QFrame, QMenuBar,
    QMenu, QToolBar, QDialog, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
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
        self.init_ui()
        
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
        self.status_bar.showMessage("Ready - Use File menu to load CSV data. Database will be auto-detected.")
        
        # Apply styling
        self.apply_styling()
        
    def delayed_apply_filters(self):
        """Apply filters with a delay to prevent crashes while typing"""
        self.filter_timer.stop()  # Stop any existing timer
        self.filter_timer.start(300)  # Wait 300ms before applying filters
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
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
            # Load database
            self.description_mapping, self.type_mapping = self.load_database(self.database_file)
            
            # Load and process CSV
            self.raw_data = self.process_csv_file(self.csv_file, self.description_mapping, self.type_mapping)
            self.group_data()
            self.apply_filters()
            self.update_display()
            
            self.export_action.setEnabled(True)
            self.status_bar.showMessage(f"Loaded {len(self.raw_data)} rows, {len(self.grouped_data)} unique items")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading files: {str(e)}")
            
    def load_database(self, database_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load database file and create mappings"""
        part_to_description = {}
        part_to_type = {}
        
        with open(database_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                part_number = row.get('Item Number', '').strip()
                description = row.get('Item Description', '').strip()
                
                # Find type column (look for "Terms" in column name)
                item_type = ""
                for key, val in row.items():
                    if "Terms" in key:
                        item_type = val
                        break
                
                if part_number and description:
                    part_to_description[part_number] = description
                    part_to_type[part_number] = item_type
                    
        return part_to_description, part_to_type
        
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
