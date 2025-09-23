# PyQt6 Lumber List Matching Results Viewer

A modern GUI application for viewing and interacting with lumber list matching results, featuring interactive SKU selection for items with multiple matches.

## Features

### 🎯 **Interactive SKU Selection**
- **ComboBox for Multiple Matches**: Items with multiple SKU matches display a dropdown listbox
- **Confidence Indicators**: Visual symbols show match confidence (✓ High, ○ Medium, ⚠ Low)
- **Real-time Updates**: Selecting a different SKU immediately updates the description, type, and confidence

### 📊 **Advanced Filtering**
- **Search**: Filter items by text search across item numbers and descriptions
- **Confidence Filter**: Set minimum confidence threshold (0-100%)
- **Match Status**: Show/hide items with or without matches
- **Real-time Filtering**: All filters apply instantly as you type or change settings

### 📋 **Data Management**
- **Grouped Display**: Items are grouped by item number, showing match counts
- **Sortable Columns**: Click column headers to sort by any field
- **CSV Export**: Export filtered results to CSV file
- **Status Bar**: Shows current item counts and application status

### 🎨 **Modern Interface**
- **Clean Design**: Professional, easy-to-read interface
- **Alternating Rows**: Better visual separation of data
- **Responsive Layout**: Resizable columns and splitter panels
- **Details Panel**: Side panel for detailed item information

## Usage

### Quick Start
```bash
# Using the launcher script (recommended)
python3 launch_pyqt6_viewer.py

# Or directly with conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jlc
python viewer_pyqt6.py
```

### Interface Overview

#### Main Table
- **Item #**: Item number from the lumber list
- **Quantity**: Quantity of the item
- **Text**: Processed description text
- **Original**: Original text from the document
- **SKU**: Part number (dropdown for multiple matches)
- **Description**: Item description from database
- **Type**: Item type from database
- **Confidence**: Match confidence with visual indicator

#### Menu Bar
- **File Menu**:
  - **Load CSV Results** (Ctrl+O): Load matching results CSV file
  - **Load Database** (Ctrl+D): Load SKU database file
  - **Export CSV** (Ctrl+E): Export current filtered results
  - **Exit** (Ctrl+Q): Close application

- **View Menu**:
  - **Show Items with Matches**: Toggle visibility of matched items
  - **Show Items with No Matches**: Toggle visibility of unmatched items

- **Help Menu**:
  - **About**: Application information

#### Toolbar
- **Search**: Text search across all fields
- **Min Confidence**: Set minimum confidence threshold (0-100%)
- **Clear Filters**: Reset all filters to default values

## File Loading

The application allows you to load different files:
- **CSV Results File**: Contains the matching results from the scanning process
- **Database File**: Contains the SKU information and descriptions

Use the File menu to load your specific files rather than relying on hardcoded paths.

## Key Features for Multiple Matches

### ComboBox Integration
When an item has multiple SKU matches:
1. The SKU column shows a dropdown (ComboBox)
2. Each option displays: `[Symbol] PartNumber`
3. Symbols indicate confidence: ✓ (≥80%), ○ (50-79%), ⚠ (<50%)
4. Selecting a different option immediately updates the description, type, and confidence columns

### Example Multiple Match Display
```
Item # | Quantity | Text           | SKU (Dropdown)     | Description
-------|----------|----------------|--------------------|------------------
31     | 18       | PCZ4           | [○] PC4Z           | Description 1
       |          |                | [✓] PC4Z-HD        | Description 2  
       |          |                | [⚠] PC4Z-STD       | Description 3
```

## Technical Details

- **Framework**: PyQt6 (Python Qt6 bindings)
- **Python Version**: 3.8+ (compatible with conda environment)
- **Dependencies**: PyQt6, standard library modules
- **Data Format**: CSV input/output
- **Architecture**: Model-View pattern with custom widgets

## Troubleshooting

### Common Issues
1. **"PyQt6 not found"**: Ensure conda environment is activated
2. **"File not found"**: Check that CSV files exist in the expected locations
3. **GUI not responsive**: Check system resources and close other applications

### Performance
- Optimized for datasets with hundreds of items
- Efficient filtering and sorting
- Memory-conscious design for large result sets

## Development

The application is built with modularity in mind:
- `SKUComboBox`: Custom widget for SKU selection
- `LumberViewerGUI`: Main application class
- Separate methods for data loading, filtering, and display updates

## Future Enhancements

Potential improvements could include:
- Drag-and-drop file loading
- Advanced search with regex support
- Batch operations on selected items
- Custom confidence thresholds per item type
- Integration with external databases
