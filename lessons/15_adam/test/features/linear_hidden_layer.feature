Feature: Linear Hidden Layer
	As a Linear Hidden Layer
	I want to be connected to network
	So that I can use it

	Rule: check weight gradient

		Scenario: check layer weight gradient
			Given create and init layer
			When design computeError and prepare next layer delta with identity activator
			And forward layer
			And compute layer delta
			And compute layer weight gradient as actual weight gradient
			And compute expect weight gradient by derivative definition
			Then expect weight gradient should equal actual weight gradient