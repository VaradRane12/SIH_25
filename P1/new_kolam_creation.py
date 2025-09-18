import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from dataclasses import dataclass
from typing import List, Tuple
import math
from io import BytesIO
import imageio

# Import the KolamGenerator classes from the previous file
# For Streamlit, we'll include them here directly

@dataclass
class KolamPattern:
    """Represents a Kolam pattern configuration"""
    name: str
    dot_rows: int
    symmetry: str
    loop_style: str
    weaving_rule: str

class KolamGenerator:
    def __init__(self, n_dots: int = 5):
        if n_dots % 2 == 0:
            n_dots += 1
        self.n_dots = n_dots
        self.dots = []
        self.curves = []
        
    def create_dot_grid(self, style='diamond'):
        self.dots = []
        
        if style == 'diamond':
            for i in range(self.n_dots):
                row_dots = self.n_dots - abs(i - self.n_dots // 2)
                y = i
                for j in range(row_dots):
                    x = j + abs(i - self.n_dots // 2) / 2
                    self.dots.append((x, y))
                    
        elif style == 'square':
            for i in range(self.n_dots):
                for j in range(self.n_dots):
                    self.dots.append((j, i))
                    
        return np.array(self.dots)
    
    def generate_loop_around_dot(self, dot, radius=0.3, n_points=100):
        x, y = dot
        t = np.linspace(0, 2*np.pi, n_points)
        loop_x = x + radius * np.cos(t)
        loop_y = y + radius * np.sin(t)
        return loop_x, loop_y
    
    def generate_kambi_kolam(self):
        curves = []
        dots = self.create_dot_grid('diamond')
        
        for i in range(len(dots)):
            dot = dots[i]
            x, y = dot
            
            neighbors = []
            for other_dot in dots:
                if np.linalg.norm(dot - other_dot) < 1.5 and not np.array_equal(dot, other_dot):
                    neighbors.append(other_dot)
            
            for neighbor in neighbors:
                curve = self._create_weaving_curve(dot, neighbor, dots)
                if curve is not None:
                    curves.append(curve)
        
        return dots, curves
    
    def _create_weaving_curve(self, dot1, dot2, all_dots, curve_tension=0.3):
        x1, y1 = dot1
        x2, y2 = dot2
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        dx = x2 - x1
        dy = y2 - y1
        
        perp_x = -dy
        perp_y = dx
        
        length = np.sqrt(perp_x**2 + perp_y**2)
        if length > 0:
            perp_x /= length
            perp_y /= length
        
        control_offset = curve_tension
        control_x = mid_x + perp_x * control_offset
        control_y = mid_y + perp_y * control_offset
        
        t = np.linspace(0, 1, 50)
        curve_x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
        curve_y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
        
        return np.column_stack([curve_x, curve_y])
    
    def generate_pulli_kolam(self, complexity='medium'):
        dots = self.create_dot_grid('diamond')
        patterns = []
        
        center_dot = dots[len(dots)//2]
        
        if complexity == 'simple':
            for dot in dots:
                loop_x, loop_y = self.generate_loop_around_dot(dot, radius=0.25)
                patterns.append(np.column_stack([loop_x, loop_y]))
                
        elif complexity == 'medium':
            for i, dot in enumerate(dots):
                x, y = dot
                same_row = [d for d in dots if abs(d[1] - y) < 0.1 and not np.array_equal(d, dot)]
                same_col = [d for d in dots if abs(d[0] - x) < 0.1 and not np.array_equal(d, dot)]
                
                for other_dot in same_row[:1]:
                    pattern = self._create_figure_eight(dot, other_dot)
                    if pattern is not None:
                        patterns.append(pattern)
                        
        elif complexity == 'complex':
            patterns = self._create_interlocking_pattern(dots)
            
        return dots, patterns
    
    def _create_figure_eight(self, dot1, dot2):
        x1, y1 = dot1
        x2, y2 = dot2
        
        t = np.linspace(0, 2*np.pi, 100)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        scale = np.linalg.norm(dot2 - dot1) / 2
        x = mid_x + scale * np.cos(t) / (1 + np.sin(t)**2)
        y = mid_y + scale * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        
        return np.column_stack([x, y])
    
    def _create_interlocking_pattern(self, dots):
        patterns = []
        
        for dot in dots:
            x, y = dot
            
            n_points = 6
            inner_radius = 0.15
            outer_radius = 0.35
            
            angles = np.linspace(0, 2*np.pi, n_points * 2 + 1)
            points = []
            
            for i, angle in enumerate(angles[:-1]):
                if i % 2 == 0:
                    r = outer_radius
                else:
                    r = inner_radius
                px = x + r * np.cos(angle)
                py = y + r * np.sin(angle)
                points.append([px, py])
            
            patterns.append(np.array(points))
            
        return patterns
    
    def draw_kolam(self, style='kambi', complexity='medium', 
                   show_dots=True, line_width=2, figsize=(10, 10),
                   line_color='black', dot_color='red'):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        if style == 'kambi':
            dots, curves = self.generate_kambi_kolam()
        else:
            dots, curves = self.generate_pulli_kolam(complexity)
        
        for curve in curves:
            if len(curve) > 0:
                ax.plot(curve[:, 0], curve[:, 1], color=line_color, 
                       linewidth=line_width)
        
        if show_dots and len(dots) > 0:
            ax.scatter(dots[:, 0], dots[:, 1], color=dot_color, s=30, zorder=5)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        if len(dots) > 0:
            margin = 1
            ax.set_xlim(dots[:, 0].min() - margin, dots[:, 0].max() + margin)
            ax.set_ylim(dots[:, 1].min() - margin, dots[:, 1].max() + margin)
        
        plt.tight_layout()
        return fig, ax
    
    def animate_kolam(self, style='kambi', complexity='medium',
                      show_dots=True, line_width=2, line_color='black', dot_color='red',
                      frames=30, interval=0.1):
        """Create an animated GIF of the Kolam being drawn progressively"""
        if style == 'kambi':
            dots, curves = self.generate_kambi_kolam()
        else:
            dots, curves = self.generate_pulli_kolam(complexity)
        
        images = []
        for step in range(1, frames + 1):
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
            
            # Draw partial curves based on animation progress
            for curve in curves:
                if len(curve) > 0:
                    n_points = int(len(curve) * step / frames)
                    if n_points > 1:
                        ax.plot(curve[:n_points, 0], curve[:n_points, 1], 
                               color=line_color, linewidth=line_width)
            
            # Always show dots if enabled
            if show_dots and len(dots) > 0:
                ax.scatter(dots[:, 0], dots[:, 1], color=dot_color, s=30, zorder=5)
            
            ax.set_aspect('equal')
            ax.axis('off')
            
            if len(dots) > 0:
                margin = 1
                ax.set_xlim(dots[:, 0].min() - margin, dots[:, 0].max() + margin)
                ax.set_ylim(dots[:, 1].min() - margin, dots[:, 1].max() + margin)
            
            plt.tight_layout()
            
            # Save frame to buffer
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            buf.seek(0)
            images.append(imageio.v2.imread(buf))
            plt.close(fig)
        
        # Create GIF
        gif_buf = BytesIO()
        imageio.mimsave(gif_buf, images, format='GIF', duration=interval)
        gif_buf.seek(0)
        return gif_buf

# Streamlit App
st.set_page_config(page_title="Kolam Pattern Generator", 
                   page_icon="🎨", 
                   layout="wide")

st.title("Mathematical Kolam Pattern Generator")
st.markdown("""
### Understanding the Mathematics Behind Traditional Indian Art
This application demonstrates the computational principles behind Kolam designs, 
showing how mathematical patterns create these beautiful traditional art forms.
""")

# Sidebar for controls
st.sidebar.header("Kolam Configuration")

# Tab layout for different modes
tab1, tab2, tab3, tab4 = st.tabs(["Custom Design", "Animation Studio", "Random Generation", "Design Principles"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Design Parameters")
        
        # Basic parameters
        n_dots = st.slider("Number of dots (center row)", 
                          min_value=3, max_value=21, value=7, step=2,
                          help="Odd numbers create symmetric patterns")
        
        style = st.selectbox("Kolam Style", 
                           options=['kambi', 'pulli'],
                           format_func=lambda x: 'Kambi (Loop)' if x == 'kambi' else 'Pulli (Dotted)')
        
        if style == 'pulli':
            complexity = st.selectbox("Pattern Complexity",
                                     options=['simple', 'medium', 'complex'])
        else:
            complexity = 'medium'
        
        st.markdown("#### Visual Settings")
        show_dots = st.checkbox("Show dots", value=True)
        line_width = st.slider("Line width", 0.5, 5.0, 2.0, 0.5)
        
        # Color selection
        col_a, col_b = st.columns(2)
        with col_a:
            line_color = st.color_picker("Line color", "#000000")
        with col_b:
            dot_color = st.color_picker("Dot color", "#FF0000")
        
        generate_btn = st.button("Generate Kolam", type="primary", use_container_width=True)
    
    with col2:
        if generate_btn or 'kolam_generated' not in st.session_state:
            with st.spinner("Creating Kolam pattern..."):
                generator = KolamGenerator(n_dots)
                fig, ax = generator.draw_kolam(
                    style=style,
                    complexity=complexity,
                    show_dots=show_dots,
                    line_width=line_width,
                    line_color=line_color,
                    dot_color=dot_color,
                    figsize=(8, 8)
                )
                
                # Display the Kolam
                st.pyplot(fig)
                
                # Save to session state
                st.session_state.kolam_generated = True
                
                # Download button
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Download Kolam",
                    data=buf,
                    file_name=f"kolam_{style}_{n_dots}dots.png",
                    mime="image/png"
                )
                
                plt.close(fig)

with tab2:
    st.subheader("🎬 Animation Studio")
    st.markdown("Create animated GIFs showing the progressive drawing of Kolam patterns.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Animation Parameters")
        
        # Animation parameters
        anim_n_dots = st.slider("Number of dots", 
                               min_value=3, max_value=15, value=7, step=2,
                               key="anim_dots")
        
        anim_style = st.selectbox("Animation Style", 
                                 options=['kambi', 'pulli'],
                                 format_func=lambda x: 'Kambi (Loop)' if x == 'kambi' else 'Pulli (Dotted)',
                                 key="anim_style")
        
        if anim_style == 'pulli':
            anim_complexity = st.selectbox("Animation Complexity",
                                          options=['simple', 'medium', 'complex'],
                                          key="anim_complexity")
        else:
            anim_complexity = 'medium'
        
        st.markdown("#### Animation Settings")
        frames = st.slider("Number of frames", 20, 60, 30, 5,
                          help="More frames = smoother animation")
        speed = st.slider("Animation speed", 0.05, 0.5, 0.1, 0.05,
                         help="Lower values = faster animation")
        
        anim_show_dots = st.checkbox("Show dots in animation", value=True, key="anim_dots_check")
        anim_line_width = st.slider("Animation line width", 1.0, 4.0, 2.0, 0.5, key="anim_line_width")
        
        # Color selection for animation
        col_a, col_b = st.columns(2)
        with col_a:
            anim_line_color = st.color_picker("Animation line color", "#000000", key="anim_line_color")
        with col_b:
            anim_dot_color = st.color_picker("Animation dot color", "#FF0000", key="anim_dot_color")
        
        create_animation_btn = st.button("Create Animation", type="primary", use_container_width=True)
    
    with col2:
        if create_animation_btn:
            with st.spinner(f"Creating animation with {frames} frames... This may take a moment."):
                try:
                    generator = KolamGenerator(anim_n_dots)
                    gif_buffer = generator.animate_kolam(
                        style=anim_style,
                        complexity=anim_complexity,
                        show_dots=anim_show_dots,
                        line_width=anim_line_width,
                        line_color=anim_line_color,
                        dot_color=anim_dot_color,
                        frames=frames,
                        interval=speed
                    )
                    
                    # Display the animation
                    st.image(gif_buffer.getvalue(), use_container_width=True)
                    
                    # Download button for GIF
                    st.download_button(
                        label="Download Animation (GIF)",
                        data=gif_buffer.getvalue(),
                        file_name=f"kolam_animation_{anim_style}_{anim_n_dots}dots.gif",
                        mime="image/gif"
                    )
                    
                    st.success("Animation created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating animation: {str(e)}")
                    st.info("Try reducing the number of frames or complexity if you encounter issues.")

with tab3:
    st.subheader("Random Kolam Generator")
    st.markdown("Generate random Kolam patterns to explore the variety of designs possible.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Random Parameters")
        min_dots = st.number_input("Minimum dots", 3, 15, 5, 2)
        max_dots = st.number_input("Maximum dots", 7, 21, 15, 2)
        
        if st.button("Generate Random Kolam", type="primary", use_container_width=True):
            # Random parameters
            n_dots = random.choice(range(min_dots, max_dots + 1, 2))
            style = random.choice(['kambi', 'pulli'])
            complexity = random.choice(['simple', 'medium', 'complex'])
            show_dots = random.choice([True, False])
            line_width = random.uniform(1.5, 3)
            
            # Store in session state
            st.session_state.random_params = {
                'n_dots': n_dots,
                'style': style,
                'complexity': complexity,
                'show_dots': show_dots,
                'line_width': line_width
            }
    
    with col2:
        if 'random_params' in st.session_state:
            params = st.session_state.random_params
            
            # Display parameters
            st.info(f"""
            **Generated Parameters:**
            - Dots: {params['n_dots']}
            - Style: {params['style'].capitalize()}
            - Complexity: {params['complexity'].capitalize()}
            - Show Dots: {params['show_dots']}
            - Line Width: {params['line_width']:.1f}
            """)
            
            # Generate and display
            generator = KolamGenerator(params['n_dots'])
            fig, ax = generator.draw_kolam(
                style=params['style'],
                complexity=params['complexity'] if params['style'] == 'pulli' else 'medium',
                show_dots=params['show_dots'],
                line_width=params['line_width'],
                figsize=(8, 8)
            )
            
            st.pyplot(fig)
            plt.close(fig)

with tab4:
    st.subheader("📚 Mathematical Design Principles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Principles
        
        **1. Symmetry**
        - Rotational symmetry (4-fold, 8-fold)
        - Reflection symmetry
        - Translation symmetry in patterns
        
        **2. Dot Grids (Pulli)**
        - Diamond/rhombic arrangements
        - Square grids
        - Triangular lattices
        
        **3. Curve Generation**
        - Continuous loops (Kambi Kolam)
        - Bezier curves for smooth paths
        - Figure-8 patterns (Lemniscate)
        """)
    
    with col2:
        st.markdown("""
        ### Mathematical Concepts
        
        **1. Graph Theory**
        - Dots as vertices
        - Curves as edges
        - Eulerian paths for continuous drawing
        
        **2. Geometric Transformations**
        - Rotation matrices
        - Reflection operations
        - Scaling transformations
        
        **3. Pattern Recognition**
        - Identifying repeating motifs
        - Fractal-like self-similarity
        - Modular arithmetic in patterns
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Algorithm Overview
    
    The generator uses several mathematical approaches:
    
    1. **Dot Grid Generation**: Creates symmetric dot arrangements using coordinate geometry
    2. **Curve Weaving**: Implements Bezier curves to create smooth paths between dots
    3. **Pattern Symmetry**: Applies transformation matrices to ensure rotational and reflective symmetry
    4. **Complexity Levels**: Uses different mathematical patterns (circles, figure-8s, stars) based on complexity
    5. **Animation**: Progressive curve drawing using frame-by-frame rendering
    
    This demonstrates how traditional art forms encode sophisticated mathematical principles that can be 
    computationally modeled and recreated.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Kolam Generator - Exploring the Mathematics of Traditional Indian Art</small>
</div>
""", unsafe_allow_html=True)