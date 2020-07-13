#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self,device,model):
        ### TODO: Initialize any class variables desired ###
        self.model_xml = model
        self.device = device
        self.IECore =IECore()
        self.execNetwork = None

        self.inputBlob = None
        self.outputBlob = None


    def load_model(self):
        ### TODO: Load the model ###

        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        try:
            self.network = self.IECore.read_network(self.model_xml, os.path.splitext(self.model_xml)[0] + ".bin")
        except AttributeError:
            self.network = IENetwork(model=self.model_xml, weights=os.path.splitext(self.model_xml)[0] + ".bin")
        
        # Loading IE network into plugin
        self.execNetwork=self.IECore.load_network(network=self.network, device_name=self.device)

        # check for supported layers
        supportedLayers = self.IECore.query_network(network=self.network, device_name = self.device)

        unsupportedLayers =[l for l in self.network.layers.keys() if l not in supportedLayers]

        if len(unsupportedLayers) != 0:
            print(unsupportedLayers)

        self.inputBlob = next(iter(self.network.inputs))
        self.outputBlob = next(iter(self.network.outputs))


        

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.inputBlob].shape

    def exec_net(self, image):
        self.execNetwork.start_async(request_id =0, inputs = {self.inputBlob:image})
        while True:
            status = self.wait()
            if status ==0:
                break
            else:
                time.sleep(-1)
        return self.execNetwork


    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.execNetwork.requests[0].wait(-1)

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.execNetwork.requests[0].outputs[self.outputBlob]
