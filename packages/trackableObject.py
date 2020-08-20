# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:57:49 2020

@author: sai seravan
"""

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.counted = False 