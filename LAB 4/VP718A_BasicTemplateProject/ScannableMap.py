import numpy as np
import pygame

class ScanableMap:
  def __init__(self,map_dimensions=(100,100),fov=360,max_distance=100,num_rays=360,scale=10,draw_unit=10,endpoint=False):
    self.map_rows, self.map_cols = map_dimensions
    grid_dimensions=(self.map_rows*scale,self.map_cols*scale)
    self.grid = np.zeros(grid_dimensions)
    self.draw_boxes=[]
    self.rows, self.cols = grid_dimensions

    self.max_distance=max_distance 
    self.num_rays=num_rays 
    self.fov=np.radians(fov)
    self.draw_unit=draw_unit 
    self.scale=scale
    self.endpoint=endpoint

  def set_solid(self,x,y):
    for a in range(self.scale):
      for b in range(self.scale):
        self.grid[y*self.scale+a][x*self.scale+b]=1
    self.draw_boxes.append( (x*self.draw_unit,y*self.draw_unit,self.draw_unit,self.draw_unit))

  def load_map(self,fname):
    with open(fname,"r") as map_file:
      lines=[line.strip() for line in map_file if line.strip()]
    if len(lines)!=self.map_rows or any(len(line)>self.map_cols for line in lines):
      raise ValueError(
        f"Map exceeds scanner dimensions: expected "
        f"{self.map_rows} rows by {self.map_cols} columns"
      )
    row=0
    # Missing trailing cells stay free, matching world.map_from_file.
    for l in lines:
      col=0
      for c in l.strip():
        if c=="1" or c=="2":
          self.set_solid(col,row)
        col+=1
      row+=1

  def draw_map(self,display_surface):
    """This assumed a PyGame draw surface"""
    for i in self.draw_boxes:
      pygame.draw.rect(display_surface, (255,255,255), i)   

  def scan(self,start_x,start_y,start_a):
    # Generate all angles at once
    # note that we take back half the fov, and add on the start angle of the robot
    angles = np.linspace(-self.fov/2+start_a, self.fov/2+start_a, self.num_rays, endpoint=self.endpoint)
    
    dx = np.cos(angles)  # Shape: (num_rays,)
    dy = np.sin(angles)

    # Create step distances array
    steps = np.arange(0, self.max_distance, 0.5)  # Shape: (num_steps,)
   
    # Broadcast to get all positions for all rays at all steps
    # Shape becomes: (num_rays, num_steps)
    x_positions = start_x + dx[:, np.newaxis] * steps[np.newaxis, :]
    y_positions = start_y + dy[:, np.newaxis] * steps[np.newaxis, :]
 
    
    # Convert to integer indices
    x_indices = np.floor(x_positions).astype(int)
    y_indices = np.floor(y_positions).astype(int)
 
    # Check bounds - create mask for valid positions
    valid = ((x_indices >= 0) & (x_indices < self.cols) & 
             (y_indices >= 0) & (y_indices < self.rows))

    # Sample grid at all positions (set invalid positions to 0)
    x_safe = np.clip(x_indices, 0, self.cols - 1)
    y_safe = np.clip(y_indices, 0, self.rows - 1)
    
    
    samples = self.grid[y_safe, x_safe]
    
    # Mark invalid positions as "no wall"
    samples = samples * valid
    
    # Find the first wall hit for each ray; map edges are open space.
    hit_mask = (samples == 1) & valid
   
    # Get index of first hit for each ray
    first_hit_idx = np.argmax(hit_mask, axis=1)
    
    # Calculate distances
    distances = steps[first_hit_idx]
    
    # Handle rays that never hit (all False in hit_mask)
    no_hit = ~np.any(hit_mask, axis=1)
    distances[no_hit] = self.max_distance
    
    return distances, angles

  def is_free(self,x,y):
    """Return whether one or more grid-coordinate positions are in free map cells."""
    x_indices=np.floor(np.asarray(x)).astype(int)
    y_indices=np.floor(np.asarray(y)).astype(int)
    valid=(x_indices>=0)&(x_indices<self.cols)&(y_indices>=0)&(y_indices<self.rows)
    safe_x=np.clip(x_indices,0,self.cols-1)
    safe_y=np.clip(y_indices,0,self.rows-1)
    return valid & (self.grid[safe_y,safe_x]==0)