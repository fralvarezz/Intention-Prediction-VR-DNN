using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;

public class Predictor : MonoBehaviour
{

    public Texture2D texture;
    
    public NNModel modelAsset;

    private Model _runtimeModel;

    private IWorker _worker;

    private Dictionary<string, Tensor> inputs;

    /// <summary>
    /// A struct used for holding the results of our prediction in a way that's easy to view from the inspector.
    /// </summary>
    [Serializable]
    public struct Prediction
    {
        // The most likely value for this prediction
        public int predictedValue;
        // The list of likelihoods for all the possible classes
        public float[] predicted;

        public void SetPrediction(Tensor t)
        {
            // Extract the float value outputs into the predicted array.
            predicted = t.AsFloats();
            // The most likely one is the predicted value.
            predictedValue = Array.IndexOf(predicted, predicted.Max());
            Debug.Log($"Predicted {predictedValue}");
        }
    }

    public Prediction prediction;
    
    void Start()
    {
        // Load the model and make a worker
        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = WorkerFactory.CreateWorker(_runtimeModel, WorkerFactory.Device.GPU);
        
        // Instantiate the prediction struct.
        prediction = new Prediction();
        inputs = new Dictionary<string, Tensor>();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // making a tensor out of a grayscale texture
            var channelCount = 1; //1 = grayscale, 3 = color, 4 = color+alpha
            // Create a tensor for input from the texture.
            var inputX = new Tensor(texture, channelCount);
            
            //Reshape tht Tensor to 1x28x28
            inputX = inputX.Reshape(new TensorShape(1,1,28,28));

            //Figured out that w and c have to be inverted. There is no tensor transpose so I do it manually here.
            Tensor newTensor = new Tensor(1, 1, 28, 28);

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    newTensor[0, 0, i, j] = inputX[0, 0, j, i];
                }
            }

            // Peek at the output tensor without copying it.
            Tensor outputY = _worker.Execute(newTensor).PeekOutput();
            
            // Set the values of our prediction struct using our output tensor.
            prediction.SetPrediction(outputY);
            
            // Dispose of the input tensor manually (not garbage-collected).
            newTensor.Dispose();
        }
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        _worker?.Dispose();
    }
    
    public static void FlipTexture(ref Texture2D texture)
    {
        int textureWidth = texture.width;
        int textureHeight = texture.height;
 
        Color32[] pixels = texture.GetPixels32();
 
        for (int y = 0; y < textureHeight; y++)
        {
            int yo = y * textureWidth;
            for (int il = yo, ir = yo + textureWidth - 1; il < ir; il++, ir--)
            {
                (pixels[il], pixels[ir]) = (pixels[ir], pixels[il]);
            }
        }
        texture.SetPixels32(pixels);
        texture.Apply();
    }
}