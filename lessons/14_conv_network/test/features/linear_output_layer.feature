Feature: Linear Output Layer
	As a Linear Output Layer
	I want to be as output of network
	So that I can use it

	Rule: check weight gradient

		Scenario: check layer weight gradient
			Given create and init layer
			When forward layer
			And compute layer delta
			And compute layer weight gradient as actual weight gradient
			And compute expect weight gradient by derivative definition
			Then expect weight gradient should equal actual weight gradient