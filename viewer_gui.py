#!/usr/bin/env python3
"""
GUI Viewer for Lumber List Matching Results

A modern graphical interface for viewing and analyzing lumber list matching results.
Features include filtering, sorting, search, and export capabilities.
"""

import csv
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

class LumberViewerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lumber List Matching Results Viewer")
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)
        
        # Data storage
        self.raw_data = []
        self.filtered_data = []
        self.database_mapping = {}
        self.type_mapping = {}
        
        # GUI state
        self.current_sort_column = None
        self.sort_reverse = False
        
        self.setup_ui()
        self.load_sample_data()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Lumber List Matching Results", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Data table
        self.setup_data_table(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
    def setup_control_panel(self, parent):
        """Set up the control panel with file loading and filtering options"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # File loading section
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.csv_file_var = tk.StringVar()
        csv_entry = ttk.Entry(file_frame, textvariable=self.csv_file_var, width=50)
        csv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_csv_file).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Database:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.database_file_var = tk.StringVar()
        db_entry = ttk.Entry(file_frame, textvariable=self.database_file_var, width=50)
        db_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_database_file).grid(row=1, column=2, pady=(5, 0))
        
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=2, column=0, pady=(10, 0))
        
        # Filtering section
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        filter_frame.columnconfigure(1, weight=1)
        
        ttk.Label(filter_frame, text="Search:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        search_entry.bind('<KeyRelease>', self.on_search_change)
        
        ttk.Label(filter_frame, text="Confidence:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.confidence_var = tk.StringVar(value="All")
        confidence_combo = ttk.Combobox(filter_frame, textvariable=self.confidence_var, 
                                      values=["All", "High", "Medium", "Low"], width=10)
        confidence_combo.grid(row=0, column=3, padx=(0, 10))
        confidence_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        ttk.Label(filter_frame, text="Has Match:").grid(row=0, column=4, sticky=tk.W, padx=(10, 5))
        self.match_var = tk.StringVar(value="All")
        match_combo = ttk.Combobox(filter_frame, textvariable=self.match_var,
                                  values=["All", "Yes", "No"], width=10)
        match_combo.grid(row=0, column=5, padx=(0, 10))
        match_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        ttk.Button(filter_frame, text="Clear Filters", command=self.clear_filters).grid(row=0, column=6)
        
    def setup_data_table(self, parent):
        """Set up the data table with treeview"""
        table_frame = ttk.LabelFrame(parent, text="Results", padding="5")
        table_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        columns = ('Item', 'Qty', 'Text', 'Original', 'SKU', 'Description', 'Type', 'Confidence')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        column_widths = {
            'Item': 60, 'Qty': 60, 'Text': 200, 'Original': 200, 
            'SKU': 100, 'Description': 300, 'Type': 80, 'Confidence': 80
        }
        
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by_column(c))
            self.tree.column(col, width=column_widths[col], minwidth=50)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Bind double-click event
        self.tree.bind('<Double-1>', self.on_item_double_click)
        
    def setup_status_bar(self, parent):
        """Set up the status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Export button
        ttk.Button(status_frame, text="Export to CSV", command=self.export_to_csv).grid(row=0, column=1, padx=(10, 0))
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Try to load from 1_results directory if it exists
        results_dir = Path("1_results")
        if results_dir.exists():
            csv_file = results_dir / "lumber_matches.csv"
            db_file = results_dir / "skulist_fixed.csv"
            
            if csv_file.exists() and db_file.exists():
                self.csv_file_var.set(str(csv_file))
                self.database_file_var.set(str(db_file))
                self.load_data()
                return
        
        # If no sample data, show empty state
        self.status_var.set("No data loaded. Please select CSV and database files.")
        
    def browse_csv_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_file_var.set(filename)
            
    def browse_database_file(self):
        """Browse for database file"""
        filename = filedialog.askopenfilename(
            title="Select Database File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.database_file_var.set(filename)
            
    def load_database(self, database_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load database file and create mappings"""
        part_to_description = {}
        part_to_type = {}
        
        try:
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
                        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading database: {e}")
            return {}, {}
            
        return part_to_description, part_to_type
        
    def load_data(self):
        """Load and process data from CSV and database files"""
        csv_file = self.csv_file_var.get()
        database_file = self.database_file_var.get()
        
        if not csv_file or not database_file:
            messagebox.showwarning("Warning", "Please select both CSV and database files.")
            return
            
        if not Path(csv_file).exists() or not Path(database_file).exists():
            messagebox.showerror("Error", "One or both files do not exist.")
            return
            
        try:
            # Load database
            self.database_mapping, self.type_mapping = self.load_database(database_file)
            
            # Load CSV data
            self.raw_data = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item_number = row.get('Item_Number', '').strip()
                    quantity = row.get('Quantity', '').strip()
                    processed_text = row.get('Description', '').strip()
                    original_text = row.get('Original_Text', '').strip()
                    part_number = row.get('Part_Number', '').strip()
                    confidence = row.get('Confidence', '').strip()
                    
                    # Get description and type from database
                    item_description = ""
                    item_type = ""
                    if part_number and part_number in self.database_mapping:
                        item_description = self.database_mapping[part_number]
                        item_type = self.type_mapping[part_number]
                    
                    self.raw_data.append({
                        'item_number': item_number,
                        'quantity': quantity,
                        'processed_text': processed_text,
                        'original_text': original_text,
                        'part_number': part_number,
                        'item_description': item_description,
                        'item_type': item_type,
                        'confidence': confidence,
                        'has_match': 'Yes' if part_number else 'No'
                    })
            
            # Apply current filters
            self.apply_filters()
            self.update_display()
            
            self.status_var.set(f"Loaded {len(self.raw_data)} items from {len(self.database_mapping)} database entries")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")
            
    def apply_filters(self):
        """Apply current filters to the data"""
        self.filtered_data = self.raw_data.copy()
        
        # Search filter
        search_term = self.search_var.get().lower()
        if search_term:
            self.filtered_data = [
                row for row in self.filtered_data
                if (search_term in str(row['processed_text']).lower() or
                    search_term in str(row['original_text']).lower() or
                    search_term in str(row['part_number']).lower() or
                    search_term in str(row['item_description']).lower())
            ]
        
        # Confidence filter
        confidence_filter = self.confidence_var.get()
        if confidence_filter != "All":
            self.filtered_data = [
                row for row in self.filtered_data
                if self.get_confidence_level(row['confidence']) == confidence_filter.lower()
            ]
        
        # Match filter
        match_filter = self.match_var.get()
        if match_filter != "All":
            self.filtered_data = [
                row for row in self.filtered_data
                if row['has_match'] == match_filter
            ]
            
    def get_confidence_level(self, confidence_str):
        """Convert confidence string to level"""
        try:
            conf_val = float(confidence_str)
            if conf_val >= 0.8:
                return "high"
            elif conf_val >= 0.5:
                return "medium"
            else:
                return "low"
        except:
            return confidence_str.lower()
            
    def on_search_change(self, event=None):
        """Handle search text change"""
        self.apply_filters()
        self.update_display()
        
    def on_filter_change(self, event=None):
        """Handle filter change"""
        self.apply_filters()
        self.update_display()
        
    def clear_filters(self):
        """Clear all filters"""
        self.search_var.set("")
        self.confidence_var.set("All")
        self.match_var.set("All")
        self.apply_filters()
        self.update_display()
        
    def sort_by_column(self, column):
        """Sort data by column"""
        if self.current_sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.current_sort_column = column
            self.sort_reverse = False
            
        # Define sort key function
        def sort_key(row):
            value = row.get(column.lower().replace(' ', '_'), '')
            try:
                return float(value)
            except:
                return str(value).lower()
                
        self.filtered_data.sort(key=sort_key, reverse=self.sort_reverse)
        self.update_display()
        
    def update_display(self):
        """Update the treeview display"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add filtered data
        previous_item = None
        for row in self.filtered_data:
            # Only show item details for new items (grouping)
            is_new_item = row['item_number'] != previous_item
            
            item_values = (
                row['item_number'] if is_new_item else '',
                row['quantity'] if is_new_item else '',
                row['processed_text'] if is_new_item else '',
                row['original_text'] if is_new_item else '',
                row['part_number'],
                row['item_description'],
                row['item_type'][:6] if row['item_type'] else '',  # Truncate type
                f"{float(row['confidence']):.2f}" if row['confidence'] else ''
            )
            
            # Color code based on confidence
            item_id = self.tree.insert('', 'end', values=item_values)
            
            # Apply color coding
            confidence_level = self.get_confidence_level(row['confidence'])
            if confidence_level == "high":
                self.tree.set(item_id, 'Confidence', f"✓ {item_values[7]}")
            elif confidence_level == "low":
                self.tree.set(item_id, 'Confidence', f"⚠ {item_values[7]}")
                
            previous_item = row['item_number']
            
        # Update status
        self.status_var.set(f"Showing {len(self.filtered_data)} of {len(self.raw_data)} items")
        
    def on_item_double_click(self, event):
        """Handle double-click on item"""
        selection = self.tree.selection()
        if not selection:
            return
            
        item = self.tree.item(selection[0])
        values = item['values']
        
        # Create detail window
        self.show_item_details(values)
        
    def show_item_details(self, values):
        """Show detailed information for an item"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Item Details")
        detail_window.geometry("600x400")
        
        # Create scrolled text widget
        text_widget = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Format details
        details = f"""
ITEM DETAILS
{'='*50}

Item Number: {values[0]}
Quantity: {values[1]}
Processed Text: {values[2]}
Original Text: {values[3]}
Part Number (SKU): {values[4]}
Item Description: {values[5]}
Type: {values[6]}
Confidence: {values[7]}

{'='*50}
        """
        
        text_widget.insert(tk.END, details)
        text_widget.config(state=tk.DISABLED)
        
    def export_to_csv(self):
        """Export filtered data to CSV"""
        if not self.filtered_data:
            messagebox.showwarning("Warning", "No data to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['item_number', 'quantity', 'processed_text', 'original_text', 
                                'part_number', 'item_description', 'item_type', 'confidence']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.filtered_data)
                    
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {e}")

def main():
    """Main function"""
    if len(sys.argv) == 3:
        # Command line mode - load specified files
        csv_file, database_file = sys.argv[1], sys.argv[2]
        if not Path(csv_file).exists() or not Path(database_file).exists():
            print(f"Error: One or both files do not exist.")
            print(f"CSV: {csv_file}")
            print(f"Database: {database_file}")
            sys.exit(1)
    elif len(sys.argv) > 1:
        print("Usage: python viewer_gui.py [csv_file database_file]")
        sys.exit(1)
    
    # Create and run GUI
    root = tk.Tk()
    app = LumberViewerGUI(root)
    
    # If files provided via command line, load them
    if len(sys.argv) == 3:
        app.csv_file_var.set(sys.argv[1])
        app.database_file_var.set(sys.argv[2])
        app.load_data()
    
    root.mainloop()

if __name__ == "__main__":
    main()