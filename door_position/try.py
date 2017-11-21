class trythis:
	""" Don't have to initialize data attributes;  they can be defined directly in method attributes.
	"""	
	attr_directly_under_class_def = 30
	def seeattr(self):
		self.attr = 20
	def seeagain(self):
		self.attr = 200
		
		
if __name__ == "__main__":
	print trythis.__doc__
	x = trythis()
	x.seeattr()
	print x.attr
	x.seeagain()
	print x.attr
	print x.attr_directly_under_class_def
