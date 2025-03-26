import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np



@dataclass
class BVH:
  primitive_idx: np.ndarray
  left: np.ndarray
  right: np.ndarray

  aabb_min: np.ndarray  
  aabb_max: np.ndarray

  @staticmethod 
  def from_df(df: pd.DataFrame):
    primitive_idx = df['primitiveIdx'].values
    left = df['left'].values
    right = df['right'].values
    aabb_min = np.array([df['aabb_min_x'].values, df['aabb_min_y'].values, df['aabb_min_z'].values]).T
    aabb_max = np.array([df['aabb_max_x'].values, df['aabb_max_y'].values, df['aabb_max_z'].values]).T

    return BVH(primitive_idx, left, right, aabb_min, aabb_max)
  
  def check_equal(self, other):
    assert self.primitive_idx.shape == other.primitive_idx.shape, \
      f"primitive_idx shapes do not match {self.primitive_idx.shape} vs {other.primitive_idx.shape}"
     
    def compare_eq(name, a, b):
      assert np.allclose(self.primitive_idx == other.primitive_idx.sum()), \
        f"{name} does not match)"
    
    compare_eq('primitive_idx', self.primitive_idx, other.primitive_idx)
    compare_eq('left', self.left, other.left)
    compare_eq('right', self.right, other.right)

    compare_eq('aabb_min', self.aabb_min, other.aabb_min)
    compare_eq('aabb_max', self.aabb_max, other.aabb_max)
     


def read_bvh_csv(csv_file:Path) -> BVH:
  df = pd.read_csv(csv_file, delimiter='\s+')
  return BVH.from_df(df)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BVH CSV files")
    parser.add_argument("file1", type=Path, help="Path to first CSV file")
    parser.add_argument("file2", type=Path, help="Path to second CSV file")
    args = parser.parse_args()
    
    bvh1 = read_bvh_csv(args.file1)
    bvh2 = read_bvh_csv(args.file2)
    
    bvh1.check_equal(bvh2)