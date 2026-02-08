"""
===============================================================================
WAVERIDER GUI - PLANFORM-CUSTOMIZED DESIGN INTEGRATION PATCH
===============================================================================

This file contains the code changes needed to add planform-customized waverider
design capability to the existing waverider_gui.py.

Required files (copy to your project directory):
1. planform_customized_generator.py - The core generator class  
2. planform_gui_components.py - GUI widget components

===============================================================================
HOW TO APPLY:
===============================================================================

The easiest approach is to use the provided function apply_changes() which
generates a modified version of your waverider_gui.py.

Alternatively, follow the manual steps below.

===============================================================================
"""

import re

def apply_changes(input_file='waverider_gui.py', output_file='waverider_gui_with_planform.py'):
    """
    Apply all necessary changes to integrate planform-customized design.
    
    Parameters
    ----------
    input_file : str
        Path to original waverider_gui.py
    output_file : str
        Path for the modified output file
    """
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 1. Add QStackedWidget and QComboBox to imports
    content = content.replace(
        'QProgressBar, QTextEdit, QFileDialog, QInputDialog)',
        'QProgressBar, QTextEdit, QFileDialog, QInputDialog,\n                             QStackedWidget, QComboBox)'
    )
    
    # 2. Add new imports after existing imports
    import_insert = '''
# Import planform-customized waverider components
try:
    from planform_customized_generator import PlanformCustomizedWaverider
    from planform_gui_components import (
        DesignMethodSelector,
        PlanformParameterPanel,
        create_planform_waverider_from_gui
    )
    PLANFORM_AVAILABLE = True
except ImportError as e:
    print(f"Planform-customized design not available: {e}")
    PLANFORM_AVAILABLE = False

'''
    
    # Insert after SURROGATE_AVAILABLE block
    content = content.replace(
        'SURROGATE_AVAILABLE = False\n\n\nclass WaveriderCanvas',
        f'SURROGATE_AVAILABLE = False\n{import_insert}\nclass WaveriderCanvas'
    )
    
    # 3. Add design_method attribute to __init__
    content = content.replace(
        'self.last_stl_file = None\n        self.init_ui()',
        "self.last_stl_file = None\n        self.design_method = 'traditional'  # 'traditional' or 'planform'\n        self.init_ui()"
    )
    
    # 4. Add the switch_design_method function and modify generate_waverider
    # Find the generate_waverider function and add the new method before it
    
    new_methods = '''
    def switch_design_method(self, index):
        """Switch between traditional and planform-customized design methods."""
        method = self.method_combo.currentData() if hasattr(self, 'method_combo') else 'traditional'
        self.design_method = method
        
        if method == 'traditional':
            self.design_param_stack.setCurrentIndex(0)
            self.method_description.setText(
                "Traditional: Control shockwave/upper surface via X1-X4 parameters"
            )
            self.geom_constraint_label.setVisible(True)
            if hasattr(self, 'design_constraint_label'):
                self.design_constraint_label.setVisible(True)
        else:
            self.design_param_stack.setCurrentIndex(1)
            self.method_description.setText(
                "Planform-Customized: Directly specify leading edge shape for vortex-lift designs"
            )
            self.geom_constraint_label.setVisible(False)
            if hasattr(self, 'design_constraint_label'):
                self.design_constraint_label.setVisible(False)

'''
    
    # Insert before generate_waverider
    content = content.replace(
        '    def generate_waverider(self):',
        new_methods + '    def generate_waverider(self):'
    )
    
    # 5. Modify generate_waverider to handle both methods
    old_generate = '''    def generate_waverider(self):
        """Generate the waverider with current parameters"""
        try:
            self.info_label.setText("Generating waverider... Please wait.")
            QApplication.processEvents()
            
            # Get parameters
            M_inf = self.m_inf_spin.value()
            beta = self.beta_spin.value()
            height = self.height_spin.value()
            width = self.width_spin.value()
            dp = [
                self.x1_spin.value(),
                self.x2_spin.value(),
                self.x3_spin.value(),
                self.x4_spin.value()
            ]
            n_planes = self.n_planes_spin.value()
            n_streamwise = self.n_streamwise_spin.value()
            delta_streamwise = self.delta_streamwise_spin.value()
            n_upper_surface = self.n_us_spin.value()
            n_shockwave = self.n_sw_spin.value()
            
            # Check design space constraint
            constraint = dp[1] / ((1 - dp[0])**4)
            max_constraint = (7/64) * (width/height)**4
            
            if constraint >= max_constraint:
                QMessageBox.warning(self, "Design Space Violation",
                    f"Design parameters violate the design space constraint!\\n\\n"
                    f"Constraint value: {constraint:.4f}\\n"
                    f"Maximum allowed: {max_constraint:.4f}\\n\\n"
                    f"Try reducing X2 or increasing X1.")
                self.info_label.setText("Design space constraint violated!")
                return
            
            # Generate waverider
            self.waverider = wr(
                M_inf=M_inf,
                beta=beta,
                height=height,
                width=width,
                dp=dp,
                n_upper_surface=n_upper_surface,
                n_shockwave=n_shockwave,
                n_planes=n_planes,
                n_streamwise=n_streamwise,
                delta_streamwise=delta_streamwise
            )'''
    
    new_generate = '''    def generate_waverider(self):
        """Generate the waverider with current parameters"""
        try:
            self.info_label.setText("Generating waverider... Please wait.")
            QApplication.processEvents()
            
            # Get common parameters
            M_inf = self.m_inf_spin.value()
            beta = self.beta_spin.value()
            height = self.height_spin.value()
            width = self.width_spin.value()
            n_planes = self.n_planes_spin.value()
            n_streamwise = self.n_streamwise_spin.value()
            delta_streamwise = self.delta_streamwise_spin.value()
            n_upper_surface = self.n_us_spin.value()
            n_shockwave = self.n_sw_spin.value()
            
            # Generate based on design method
            if self.design_method == 'planform' and PLANFORM_AVAILABLE:
                # Planform-customized method
                planform_params = self.planform_panel.get_planform_params()
                mesh_params = {
                    'n_planes': n_planes,
                    'n_streamwise': n_streamwise,
                    'delta_streamwise': delta_streamwise,
                    'n_upper_surface': n_upper_surface,
                    'n_shockwave': n_shockwave
                }
                
                self.waverider = create_planform_waverider_from_gui(
                    M_inf=M_inf,
                    beta=beta,
                    height=height,
                    width=width,
                    planform_params=planform_params,
                    mesh_params=mesh_params
                )
                
                # Update info for planform method
                pf_type = planform_params.get('type', 'unknown')
                self.info_label.setText(
                    f"✓ Planform-customized waverider generated!\\n\\n"
                    f"Planform type: {pf_type}\\n"
                    f"Length: {self.waverider.length:.3f} m\\n"
                    f"Width: {width:.3f} m\\n"
                    f"Height: {height:.3f} m"
                )
            else:
                # Traditional method
                dp = [
                    self.x1_spin.value(),
                    self.x2_spin.value(),
                    self.x3_spin.value(),
                    self.x4_spin.value()
                ]
                
                # Check design space constraint
                constraint = dp[1] / ((1 - dp[0])**4)
                max_constraint = (7/64) * (width/height)**4
                
                if constraint >= max_constraint:
                    QMessageBox.warning(self, "Design Space Violation",
                        f"Design parameters violate the design space constraint!\\n\\n"
                        f"Constraint value: {constraint:.4f}\\n"
                        f"Maximum allowed: {max_constraint:.4f}\\n\\n"
                        f"Try reducing X2 or increasing X1.")
                    self.info_label.setText("Design space constraint violated!")
                    return
                
                # Generate waverider
                self.waverider = wr(
                    M_inf=M_inf,
                    beta=beta,
                    height=height,
                    width=width,
                    dp=dp,
                    n_upper_surface=n_upper_surface,
                    n_shockwave=n_shockwave,
                    n_planes=n_planes,
                    n_streamwise=n_streamwise,
                    delta_streamwise=delta_streamwise
                )
                
                # Update info for traditional method
                length = self.waverider.length
                self.info_label.setText(
                    f"✓ Waverider generated successfully!\\n\\n"
                    f"Length: {length:.3f} m\\n"
                    f"Width: {width:.3f} m\\n"
                    f"Height: {height:.3f} m\\n"
                    f"Constraint: {constraint:.4f} / {max_constraint:.4f}\\n"
                    f"Design Point: [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}, {dp[3]:.3f}]"
                )'''
    
    content = content.replace(old_generate, new_generate)
    
    # Remove the duplicate info_label update at the end of generate_waverider
    content = content.replace(
        '''            # Update all views
            self.update_all_views()
            
            # Calculate some properties
            length = self.waverider.length
            volume_approx = "N/A"  # Would need integration
            
            self.info_label.setText(
                f"✓ Waverider generated successfully!\\n\\n"
                f"Length: {length:.3f} m\\n"
                f"Width: {width:.3f} m\\n"
                f"Height: {height:.3f} m\\n"
                f"Constraint: {constraint:.4f} / {max_constraint:.4f}\\n"
                f"Design Point: [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}, {dp[3]:.3f}]"
            )''',
        '''            # Update all views
            self.update_all_views()'''
    )
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Modified GUI saved to: {output_file}")
    return output_file


# ============================================================================
# MANUAL INTEGRATION STEPS
# ============================================================================

MANUAL_STEPS = """
===============================================================================
MANUAL INTEGRATION STEPS
===============================================================================

If the automatic patch doesn't work, follow these manual steps:

1. ADD TO IMPORTS (after line 14):
   Add 'QStackedWidget, QComboBox' to the PyQt5.QtWidgets import

2. ADD NEW IMPORTS (after line 60, after SURROGATE_AVAILABLE):
   
   # Import planform-customized waverider components
   try:
       from planform_customized_generator import PlanformCustomizedWaverider
       from planform_gui_components import (
           DesignMethodSelector,
           PlanformParameterPanel, 
           create_planform_waverider_from_gui
       )
       PLANFORM_AVAILABLE = True
   except ImportError as e:
       print(f"Planform-customized design not available: {e}")
       PLANFORM_AVAILABLE = False

3. ADD TO __init__ (around line 555):
   Add: self.design_method = 'traditional'

4. MODIFY create_parameter_panel():
   - After the Geometry group, add design method selector
   - Wrap the Design Parameters group in a QStackedWidget
   - Add PlanformParameterPanel as second page of stack

5. ADD switch_design_method() function:
   See the switch_design_method code in apply_changes()

6. MODIFY generate_waverider():
   Add conditional logic to handle both 'traditional' and 'planform' methods
   See the new_generate code in apply_changes()

===============================================================================
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--apply':
        input_file = sys.argv[2] if len(sys.argv) > 2 else 'waverider_gui.py'
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'waverider_gui_with_planform.py'
        apply_changes(input_file, output_file)
    else:
        print(__doc__)
        print(MANUAL_STEPS)
        print("\nTo apply changes automatically, run:")
        print("  python integration_patch.py --apply [input_file] [output_file]")
