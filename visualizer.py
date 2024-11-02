import matplotlib.pyplot as plt
import numpy as np

class NeuralNetworkVisualizer:
    def __init__(self, input_dim, hidden_dim, output_dim, figsize=(15, 10)):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create figure and axis
        plt.style.use('dark_background')  # Better contrast for weights
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.suptitle('Neural Network Architecture', fontsize=16, color='white')
        
        # Adjust node sampling and spacing
        self.input_nodes = 20    # How many input nodes to show
        self.hidden_nodes = 15   # How many hidden nodes to show
        self.output_nodes = 10   # How many output nodes to show
        
        # Calculate node positions
        self.layer_spacing = 4   # Increased horizontal spacing
        self.vertical_spacing = 1.5  # Increased vertical spacing
        self.node_positions = self._calculate_node_positions()
        
        # Initialize plot elements
        self.weight_lines = {'input_hidden': [], 'hidden_output': []}
        self.nodes = []
        self.texts = []
        
        # Setup the plot
        self._setup_plot()
        
        # Add colorbar
        self.setup_colorbar()
        
    def _calculate_node_positions(self):
        positions = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        # Calculate positions with more spacing
        for i in range(self.input_nodes):
            y = (i - self.input_nodes/2) * self.vertical_spacing
            positions['input'].append((0, y))
            
        for i in range(self.hidden_nodes):
            y = (i - self.hidden_nodes/2) * self.vertical_spacing
            positions['hidden'].append((self.layer_spacing, y))
            
        for i in range(self.output_nodes):
            y = (i - self.output_nodes/2) * self.vertical_spacing
            positions['output'].append((2 * self.layer_spacing, y))
            
        return positions
    
    def _setup_plot(self):
        # Clear previous plot
        self.ax.clear()
        
        # Draw nodes with larger radius
        node_radius = 0.2
        
        # Draw input layer nodes
        for i, pos in enumerate(self.node_positions['input']):
            circle = plt.Circle(pos, node_radius, color='skyblue', fill=True)
            self.ax.add_artist(circle)
            self.nodes.append(circle)
            # Add node number
            if i == 0:
                self.ax.text(pos[0]-0.5, pos[1], f'1', fontsize=8, color='white')
            elif i == len(self.node_positions['input'])-1:
                self.ax.text(pos[0]-0.5, pos[1], f'{self.input_dim}', fontsize=8, color='white')
            
        # Draw hidden layer nodes
        for i, pos in enumerate(self.node_positions['hidden']):
            circle = plt.Circle(pos, node_radius, color='lightgreen', fill=True)
            self.ax.add_artist(circle)
            self.nodes.append(circle)
            # Add node number
            if i == 0:
                self.ax.text(pos[0]-0.5, pos[1], f'1', fontsize=8, color='white')
            elif i == len(self.node_positions['hidden'])-1:
                self.ax.text(pos[0]-0.5, pos[1], f'{self.hidden_dim}', fontsize=8, color='white')
            
        # Draw output layer nodes
        for i, pos in enumerate(self.node_positions['output']):
            circle = plt.Circle(pos, node_radius, color='salmon', fill=True)
            self.ax.add_artist(circle)
            self.nodes.append(circle)
            self.ax.text(pos[0]+0.5, pos[1], f'{i}', fontsize=8, color='white')
        
        # Add layer labels with larger font
        self.ax.text(0, self.vertical_spacing * (self.input_nodes/2 + 1), 
                    f'Input Layer\n({self.input_dim} nodes)', 
                    fontsize=12, ha='center', color='white')
        self.ax.text(self.layer_spacing, self.vertical_spacing * (self.hidden_nodes/2 + 1), 
                    f'Hidden Layer\n({self.hidden_dim} nodes)', 
                    fontsize=12, ha='center', color='white')
        self.ax.text(2 * self.layer_spacing, self.vertical_spacing * (self.output_nodes/2 + 1), 
                    'Output Layer\n(10 nodes)', 
                    fontsize=12, ha='center', color='white')
        
        # Set plot limits with more space
        self.ax.set_xlim(-2, 2.5 * self.layer_spacing)
        self.ax.set_ylim(-self.vertical_spacing * (max(self.input_nodes, self.hidden_nodes, self.output_nodes)/2 + 1),
                        self.vertical_spacing * (max(self.input_nodes, self.hidden_nodes, self.output_nodes)/2 + 1.5))
        self.ax.set_aspect('equal')
        
        # Remove axes
        self.ax.axis('off')
    
    def setup_colorbar(self):
        # Create a separate axes for the colorbar
        cax = self.fig.add_axes([0.95, 0.2, 0.02, 0.6])
        
        # Create color normalization
        norm = plt.Normalize(-1, 1)
        
        # Create colorbar with white labels
        self.colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
            cax=cax,
            label='Weight Value'
        )
        self.colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')
        self.colorbar.set_label('Weight Value', color='white')
    
    def update_weights(self, weights1, weights2):
        """Update the visualization with new weights"""
        # Remove old weight lines
        for line in self.ax.lines:
            line.remove()
        
        # Sample weights for visualization
        input_indices = np.linspace(0, weights1.shape[1]-1, self.input_nodes, dtype=int)
        hidden_indices = np.linspace(0, weights1.shape[0]-1, self.hidden_nodes, dtype=int)
        
        weights1_sample = weights1[hidden_indices][:, input_indices]
        weights2_sample = weights2[:, hidden_indices]
        
        # Normalize weights for better visualization
        max_weight = max(np.abs(weights1).max(), np.abs(weights2).max())
        weights1_normalized = weights1_sample / max_weight
        weights2_normalized = weights2_sample / max_weight
        
        # Draw connections between input and hidden layer
        for i, pos1 in enumerate(self.node_positions['input']):
            for j, pos2 in enumerate(self.node_positions['hidden']):
                weight = weights1_normalized[j, i]
                # Use coolwarm colormap for better visibility
                color = plt.cm.coolwarm((weight + 1) / 2)
                alpha = min(abs(weight) + 0.3, 1.0)  # Minimum opacity of 0.3
                linewidth = abs(weight) * 2 + 0.5    # Minimum linewidth of 0.5
                self.ax.plot(
                    [pos1[0], pos2[0]], 
                    [pos1[1], pos2[1]], 
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=1  # Put weights behind nodes
                )
        
        # Draw connections between hidden and output layer
        for i, pos1 in enumerate(self.node_positions['hidden']):
            for j, pos2 in enumerate(self.node_positions['output']):
                weight = weights2_normalized[j, i]
                color = plt.cm.coolwarm((weight + 1) / 2)
                alpha = min(abs(weight) + 0.3, 1.0)
                linewidth = abs(weight) * 2 + 0.5
                self.ax.plot(
                    [pos1[0], pos2[0]], 
                    [pos1[1], pos2[1]], 
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=1
                )
        
        # Refresh the plot
        self.fig.canvas.draw()
        
    def show(self):
        """Display the visualization"""
        plt.show()