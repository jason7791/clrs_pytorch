import unittest
import torch
import math
from clrs_pytorch._src.encoders import construct_encoders, _encode_inputs, initialise_encoder_on_first_pass
import torch.nn as nn
from clrs_pytorch._src.probing import DataPoint
from clrs_pytorch._src.specs import Location, Type

class TestEncoderInitialization(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64
        self.stage = "HINT"
        self.loc = "NODE"
        self.t = "SCALAR"
        self.name = "test_encoder"
        self.xavier_encoders = construct_encoders(self.stage, self.loc, self.t, self.hidden_dim, "xavier_on_scalars", self.name)
        self.default_encoders = construct_encoders(self.stage, self.loc, self.t, self.hidden_dim, "default", self.name)

    def test_xavier_encoder_initialization(self):
        """Test that LazyLinear is initialized properly on first pass for Xavier initialization."""
        input_data = torch.randn(3, 128)  # Dynamic input size
        dp = DataPoint(name="test_dp", location=Location.NODE, type_=Type.SCALAR, data=input_data)
        encoder = initialise_encoder_on_first_pass(self.xavier_encoders[0], dp.data)
        
        # Ensure weights are initialized
        self.assertIsNotNone(encoder.weight)
        self.assertIsNotNone(encoder.bias)
        
        # Check if Xavier initialization was applied correctly
        self.assertAlmostEqual(encoder.weight.mean().item(), 0.0, places=1)
        self.assertAlmostEqual(encoder.weight.std().item(), 1.0 / math.sqrt(self.hidden_dim), places=1)

    def test_default_encoder_initialization(self):
        """Test that LazyLinear initializes properly without Xavier initialization."""
        input_data = torch.randn(3, 128)  # Dynamic input size
        dp = DataPoint(name="test_dp", location=Location.NODE, type_=Type.SCALAR, data=input_data)
        encoder = initialise_encoder_on_first_pass(self.default_encoders[0], dp.data)
        
        # Ensure weights are initialized
        self.assertIsNotNone(encoder.weight)
        self.assertIsNotNone(encoder.bias)
        
        # Check that weights are not forced to Xavier initialization
        self.assertNotAlmostEqual(encoder.weight.std().item(), 1.0 / math.sqrt(self.hidden_dim), places=1)
    
    def test_encode_inputs(self):
        """Test that _encode_inputs processes data correctly after initialization."""
        input_data = torch.randn(3, 128)
        dp = DataPoint(name="test_dp", location=Location.NODE, type_=Type.SCALAR, data=input_data)
        output = _encode_inputs(self.xavier_encoders, dp)
        print(output.shape)
        # Check output shape
        self.assertEqual(output.shape[0], input_data.shape[0])
        self.assertEqual(output.shape[-1], self.hidden_dim)
    
    def test_no_reinitialization_on_second_pass(self):
        """Test that weights are not reinitialized on second pass."""
        input_data = torch.randn(3, 128)
        dp = DataPoint(name="test_dp", location=Location.NODE, type_=Type.SCALAR, data=input_data)
        
        # First pass - initialize encoder
        encoder = initialise_encoder_on_first_pass(self.xavier_encoders[0], dp.data)
        initial_weights = encoder.weight.clone().detach()
        initial_bias = encoder.bias.clone().detach()
        
        # Second pass - should not change weights
        encoder = initialise_encoder_on_first_pass(self.xavier_encoders[0], dp.data)
        
        self.assertTrue(torch.equal(initial_weights, encoder.weight))
        self.assertTrue(torch.equal(initial_bias, encoder.bias))

if __name__ == "__main__":
    unittest.main()
