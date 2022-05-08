using System;
using System.Collections;
using System.Collections.Generic;
using _Scripts;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class NetworkTestScript : MonoBehaviour
{
    // Start is called before the first frame update
    private SocketInterface socketInterface;

    private int pred;
    private string debugLogString = "Latest predicted:";
    private string debugTimestampString = "Received at:";
    private string connected = "CONNECTED";
    private string disconnected = "DISCONNECTED";
    private string STATUS = "";
    private GUIStyle currentStyle = null;
    
    private List<float> minVals = new List<float>()
    {
        -0.8447f,
        0.3382f,
        -0.3764f,
        -2.92f,
        -1.3614f,
        -0.358f,
        -0.9971f,
        -0.6687f,
        -0.9999f,
        -1.0f,
        -1.0f,
        -1.0f,
        -5.15f,
        -0.4183f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        -1.5f
    };
    
    private List<float> maxVals = new List<float>()
    {
        0.5001f, 
        1.0f, 
        0.6271f, 
        0.4839f, 
        0.3688f, 
        1.23f, 
        0.7585f, 
        1.0f, 
        0.2298f,
        0.5054f, 
        0.423f, 
        1.0f,
        0.0f, 
        1.7237f,
        20.0f, 
        9.0f, 
        2204.172f, 
        83815.45f, 
        18.8792f
    };

    public NNModel modelAsset;

    private float Normalize(float val, float min, float max)
    {
        return (val - min) / (max - min);
    }
    
    void Start()
    {
        socketInterface = gameObject.AddComponent<SocketInterface>();
        
        socketInterface.Connect("localhost", 18500);
        StartCoroutine(SimulateTraffic());
        //PredictFromCSV();
    }

    public void PredictFromCSV()
    {
        var runtimeModel = ModelLoader.Load(modelAsset);
        var worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);

        // Instantiate the prediction struct.
        var prediction = new IntentionPredictor.Prediction();
        
        var data = CSVParser.ParseCSV();

        int numCorrect = 0;
        int numTotal = 0;
        foreach (var output in data)
        {
            if(output.Count == 0)
                continue;
            for (int i = 0; i < 2; i++)
            {
                var segment = output[i];
                int startingFrame = 0;
                int endingFrame = startingFrame + 45;
                float[,] input = new float[45, segment.GetLength(1) - 1];
                while (endingFrame < segment.GetLength(0))
                {
                    for (int j = startingFrame; j < endingFrame; j++)
                    {
                        for (int k = 0; k < segment.GetLength(1) - 1; k++)
                        {
                            input[j - startingFrame, k] = segment[j, k];
                        }
                    }

                    for (int j = 0; j < input.GetLength(0); j++)
                    {
                        for (int k = 0; k < input.GetLength(1); k++)
                        {
                            input[j, k] = Normalize(input[j, k], minVals[k], maxVals[k]);
                        }
                    }
                    Debug.Log(input);
                    Tensor inputTensor = new Tensor(new TensorShape(1,1,19,45), input);

                    Tensor outputTensor = worker.Execute(inputTensor).PeekOutput();
                    prediction.SetPrediction(outputTensor);
                    Debug.Log("Predicted " + prediction.predictedValue + ". Actual: " + segment[endingFrame, segment.GetLength(1) - 1]);
                    if(prediction.predictedValue == (int)segment[endingFrame, segment.GetLength(1) - 1])
                    {
                        numCorrect++;
                    }
                    numTotal++;

                    startingFrame += 1;
                    endingFrame = startingFrame + 45;
                    inputTensor.Dispose();
                    outputTensor.Dispose();
                }
            }
        }
        Debug.Log("Num correct: " + numCorrect + " out of " + numTotal);
        Debug.Log("Accuracy: " + ((float)numCorrect / (float)numTotal) * 100 + "%");
        
    }
    
    
    // Update is called once per frame
    void Update()
    {
        //socketInterface.SendString("Test Msg");
        if (socketInterface.IsConnected())
        {
            STATUS = connected;
        }
        else
        {
            STATUS = disconnected;
        }
    }

    IEnumerator SimulateTraffic()
    {
        var data = CSVParser.ParseCSV();
        int cnt = 0;
        foreach (var output in data)
        {
            foreach (var segment in output)
            {
                for (int i = 0; i < segment.GetLength(0); i++)
                {
                    var frame = new float[segment.GetLength(1)];
                    
                    
                    for (int j = 0; j < segment.GetLength(1); j++)
                    {
                        if (j < segment.GetLength(1) - 1)
                        {
                            frame[j] = Normalize(segment[i, j], minVals[j], maxVals[j]);
                        }
                        else
                        {
                            frame[j] = segment[i, j];
                        }
                    }
                    socketInterface.SendFrame(frame);
                    yield return new WaitForSeconds(0.033f);
                    cnt++;
                    //Debug.Log("Sent " + cnt + " reqs");
                    //Debug.Log("Sent " + cnt + " reqs");
                }
            }
        }
    }
    
    private void OnGUI()
    {
        InitStyles();
        var statusText = STATUS + "\n";
        bool foundFirst = socketInterface.getPredition() != -9999;
        statusText += debugLogString + (foundFirst ? socketInterface.getPredition().ToString() : "N/A") + "\n";
        statusText += debugTimestampString + (foundFirst ? socketInterface.getTimestamp().ToString("HH:mm:ss\\Z") : "N/A") + "\n";
        GUI.Label(new Rect(10, 10, 250, 60), statusText, currentStyle);
    }


    private void InitStyles()
    {
        if( currentStyle == null )
        {
            currentStyle = new GUIStyle( GUI.skin.box );
            currentStyle.normal.background = MakeTex( 1, 1, new Color( 0f, 0f, 0f, 0.5f ) );
        }
    }
    
    private Texture2D MakeTex( int width, int height, Color col )
    {
        Color[] pix = new Color[width * height];
        for( int i = 0; i < pix.Length; ++i )
        {
            pix[ i ] = col;
        }
        Texture2D result = new Texture2D( width, height );
        result.SetPixels( pix );
        result.Apply();
        return result;
    }
    
    public void SendFrameNetwork()
    {
        var data = CSVParser.ParseCSV();
        int cnt = 0;
        foreach (var output in data)
        {
            foreach (var segment in output)
            {
                for (int i = 0; i < segment.GetLength(0); i++)
                {
                    var frame = new float[segment.GetLength(1)];
                    
                    
                    for (int j = 0; j < segment.GetLength(1); j++)
                    {
                        if (j < segment.GetLength(1) - 1)
                        {
                            frame[j] = Normalize(segment[i, j], minVals[j], maxVals[j]);
                        }
                        else
                        {
                            frame[j] = segment[i, j];
                        }
                    }
                    socketInterface.SendFrame(frame);
                    cnt++;
                    //Debug.Log("Sent " + cnt + " reqs");
                }
            }
        }
    }
    
}
