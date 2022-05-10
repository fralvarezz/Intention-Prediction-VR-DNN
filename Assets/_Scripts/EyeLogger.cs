using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;
using Valve.VR;

public class EyeLogger : MonoBehaviour
{
    public enum LoggerState
    {
        Inference,
        Logging
    }

    public LoggerState state;
    
    // Stream Writer variables:
    private StreamWriter writer;
    private int currentFrame = 0;
    private DateTime time;
    
    //Player
    public GameObject player;
    private Vector3 playerPosition;
    private Quaternion playerRotation;
    
    //Right Hand
    public GameObject rightHand;
    private Vector3 rightHandPosition;
    private Quaternion rightHandRotation;
    
    //Left Hand
    public GameObject leftHand;
    private Vector3 leftHandPosition;
    private Quaternion leftHandRotation;
    
    //Gaze
    private Vector3 gazeVector;
    private Vector3 gazePoint;
    private string gazeObjectTag = "";

    public string correctLabel = "";
    
    private static EyeLogger _instance;

    [SerializeField]
    private bool USE_NORM = true;

    [SerializeField] 
    private int KEEP_EVERY = 3;

    [SerializeField] private int SEQ_LEN = 45;

    public Dictionary<String, int> nameToIntDict = new Dictionary<string, int>()
    {
        {"None", 0},
        {"WallShelf", 0},
        {"", 0},
        {" ", 0},
        {"Wall", 0},
        {"Poles", 1},
        {"Helmet", 2},
        {"Backpack", 3},
        {"Phone", 4},
        {"Float Ring", 5},
        {"Glasses", 6},
        {"Headphones", 7},
        {"Snowboard", 8},
        {"Bottle", 9}
    };

    public static EyeLogger Instance => _instance;

    private int logIndex;

    public int sequenceLength;
    public int inputLength;
    private float[] frame;

    public bool dataIsReady;
    private int capturedFrames;
    
    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        } else {
            _instance = this;
        }
    }
    
    void Start()
    {
        //state = LoggerState.Inference;
        capturedFrames = 0;
        dataIsReady = false;
        frame = new float[inputLength];
        
        logIndex = PlayerPrefs.GetInt("Index", 0);
        logIndex++;
        PlayerPrefs.SetInt("Index", logIndex);
        currentFrame = 0;
        
        writer = new StreamWriter(GetPath());

        string csvHeader =
            "frame" + DELIM +
            
            "time" + DELIM +
            
            "player_pos_x" + DELIM +
            "player_pos_y" + DELIM +
            "player_pos_z" + DELIM +
            
            "player_up_x" + DELIM +
            "player_up_y" + DELIM +
            "player_up_z" + DELIM +
            
            "player_fwd_x" + DELIM +
            "player_fwd_y" + DELIM +
            "player_fwd_z" + DELIM +
            
            "rel_r_hand_x" + DELIM +
            "rel_r_hand_y" + DELIM +
            "rel_r_hand_z" + DELIM +
            
            "r_hand_up_x" + DELIM +
            "r_hand_up_y" + DELIM +
            "r_hand_up_z" + DELIM +
            
            "r_hand_fwd_x" + DELIM +
            "r_hand_fwd_y" + DELIM +
            "r_hand_fwd_z" + DELIM +
            
            "gaze_vec_x" + DELIM +
            "gaze_vec_y" + DELIM +
            "gaze_vec_z" + DELIM +
            
            "gaze_p_x" + DELIM +
            "gaze_p_y" + DELIM +
            "gaze_p_z" + DELIM +
            
            "obj_tag" + DELIM +
            
            "gaze_to_screen_x" + DELIM +
            "gaze_to_screen_y" + DELIM +
            
            "correct_label";
        
        if(state == LoggerState.Logging)
            writer.WriteLine(csvHeader);
        
    }
    
    private void OnApplicationQuit()
    {
        writer.Close();
    }
    
    void Update()
    {
        UpdateValues();
        CaptureInferenceData();

        /*if (!dataIsReady && loggedDataQueue.Count >= SEQ_LEN)
        {
            dataIsReady = true;
        }*/
        
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (state == LoggerState.Inference)
            {
                currentFrame = 0;
                state = LoggerState.Logging;
                Debug.Log("Logging started!");
            }
            else
            {
                state = LoggerState.Inference;
                Debug.Log("Logging finished! Switching to inference...");
            }
        }

        currentFrame++;
    }

    public void Log(Vector3 gazeVec, Vector3 gazePt, string gazeObj)
    {
        gazeVector = gazeVec;
        gazePoint = gazePt;
        gazeObjectTag = gazeObj;
        
        if(state != LoggerState.Logging)
            return;

        writer.WriteLine(GetLogAsString());
        
        if(ExperimentManager.instance == null || !ExperimentManager.instance.isGuided)
            correctLabel = "";

    }

    private void UpdateValues()
    {
        time = DateTime.Now;
        
        playerPosition = player.transform.position;
        playerRotation = player.transform.rotation;

        rightHandPosition = rightHand.transform.position;
        rightHandRotation = rightHand.transform.rotation;
        
        leftHandPosition = leftHand.transform.position;
        leftHandRotation = leftHand.transform.rotation;
    }

    /*private void AddFrameToQueue(List<float> frame)
    {
        loggedDataQueue.Enqueue(frame);
    }*/

    /*private void CaptureInferenceDataQueue()
    {
        if(state != LoggerState.Inference)
            return;
        
        loggedDataQueue.Dequeue();

        Vector3 playerUp = player.transform.up;
        Vector3 playerForward = player.transform.forward;
        Vector3 rightHandPos = RELATIVE_POS ? GetRelativePosition(player.transform, rightHandPosition) : rightHandPosition;
        Vector3 rightHandUp = rightHand.transform.up;
        Vector3 rightHandForward = rightHand.transform.forward;
        Vector3 pixelPositionObject = Camera.main.WorldToScreenPoint(gazePoint);
        
        List<float> newFrame = new List<float>()
        {
            Normalize(playerUp.x, minVals[0], maxVals[0]),
            Normalize(playerUp.y, minVals[1], maxVals[1]),
            Normalize(playerUp.z, minVals[2], maxVals[2]),
            Normalize(rightHandPos.x, minVals[3], maxVals[3]),
            Normalize(rightHandPos.y, minVals[4], maxVals[4]),
            Normalize(rightHandPos.z, minVals[5], maxVals[5]),
            Normalize(rightHandUp.x, minVals[6], maxVals[6]),
            Normalize(rightHandUp.y, minVals[7], maxVals[7]),
            Normalize(rightHandUp.z, minVals[8], maxVals[8]),
            Normalize(gazeVector.x, minVals[9], maxVals[9]),
            Normalize(gazeVector.y, minVals[10], maxVals[10]),
            Normalize(gazeVector.z, minVals[11], maxVals[11]),
            Normalize(gazePoint.x, minVals[12], maxVals[12]),
            Normalize(gazePoint.y, minVals[13], maxVals[13]),
            Normalize(gazePoint.z, minVals[14], maxVals[14]),
            Normalize(TagToInt(gazeObjectTag), minVals[15], maxVals[15]),
            Normalize(pixelPositionObject.x, minVals[16], maxVals[16]),
            Normalize(pixelPositionObject.y, minVals[17], maxVals[17]),
            Normalize(pixelPositionObject.z, minVals[18], maxVals[18])
        };
        loggedDataQueue.Enqueue(newFrame);

    }*/
    
    
    private void CaptureInferenceData()
    {
        if(state != LoggerState.Inference)
            return;

        Vector3 playerUp = player.transform.up;
        Vector3 playerForward = player.transform.forward;
        Vector3 rightHandPos = RELATIVE_POS ? GetRelativePosition(player.transform, rightHandPosition) : rightHandPosition;
        Vector3 rightHandUp = rightHand.transform.up;
        Vector3 rightHandForward = rightHand.transform.forward;
        Vector3 viewPortPositionObject = Camera.main.WorldToViewportPoint(gazePoint);

        float[] newFrame = new[]
        {
            playerUp.x,
            playerUp.y,
            playerUp.z,
            playerForward.x,
            playerForward.y,
            playerForward.z,
            rightHandPos.x,
            rightHandPos.y,
            rightHandPos.z,
            rightHandUp.x,
            rightHandUp.y,
            rightHandUp.z,
            rightHandForward.x,
            rightHandForward.y,
            rightHandForward.z,
            gazeVector.x,
            gazeVector.y,
            gazeVector.z,
            gazePoint.x,
            gazePoint.y,
            gazePoint.z,
            TagToInt(gazeObjectTag),
            viewPortPositionObject.x,
            viewPortPositionObject.y
        };

        frame = newFrame;
    }

    public float Normalize(float val, float min, float max)
    {
        return (val - min) / (max - min);
    }
    
    
    private Vector3 GetRelativePosition(Transform origin, Vector3 position)
    {
        return position - origin.position;
    }

    private string VecToStr(Vector3 vec)
    {
        return vec.x.ToString("N4") + DELIM + vec.y.ToString("N4") + DELIM + vec.z.ToString("N4") + DELIM;
    }

    private string Vec2ToStr(Vector3 vec)
    {
        return vec.x.ToString("N4") + DELIM + vec.y.ToString("N4") + DELIM;
    }

    private static bool RELATIVE_POS = true;
    private static string DELIM = ";";
    
    string GetLogAsString()
    {
        string output = String.Empty;
        //If we include the timestamp then uncomment below
        output += currentFrame + DELIM + time + " " + time.Millisecond + DELIM;
        output += VecToStr(playerPosition);
        output += VecToStr(player.transform.up);
        output += VecToStr(player.transform.forward);
        
        //We focus only on right hand maybe?
        if (RELATIVE_POS)
            output += VecToStr(GetRelativePosition(player.transform, rightHandPosition));
        else
            output += VecToStr(rightHandPosition);
        
        output += VecToStr(rightHand.transform.up);
        output += VecToStr(rightHand.transform.forward);
        
        //TODO: Figure out if rightHandRotation needs to be relative or not
        output += VecToStr(gazeVector) + VecToStr(gazePoint) + TagToInt(gazeObjectTag) + DELIM;
        output += Vec2ToStr(Camera.main.WorldToViewportPoint(gazePoint));
        output += TagToInt(correctLabel); // save when a player interacts with an object, not fed to the NN
        
        return output;

    }
    
    private string GetPath()
    {
#if UNITY_EDITOR
        return Application.dataPath + $"/eyeLog_{logIndex}.csv";
#endif
    }

    private int TagToInt(string objectTag)
    {
        if(objectTag == null)
            throw new Exception("TagToInt: objectTag is null");
        
        if (!nameToIntDict.ContainsKey(objectTag))
            throw new Exception($"{objectTag} not available in tags dictionary");

        return nameToIntDict[objectTag];
    }

    public void SetInteractedObject(string objectTag)
    {
        Debug.Log("Setting interacted object: " + objectTag);
        correctLabel = objectTag;
    }

    // this worked for one frame, not for batches
    public float[,] GetVectorData()
    {
        Vector3 playerUp = player.transform.up;
        
        Vector3 rightHandPos = RELATIVE_POS ? GetRelativePosition(player.transform, rightHandPosition) : rightHandPosition;
        Vector3 rightHandUp = rightHand.transform.up;
        
        Vector3 pixelPositionObject = Camera.main.WorldToScreenPoint(gazePoint);
        return new float[,]
        {
            {
                playerUp.x,
                playerUp.y,
                playerUp.z,
                rightHandPos.x,
                rightHandPos.y,
                rightHandPos.z,
                rightHandUp.x,
                rightHandUp.y,
                rightHandUp.z,
                gazeVector.x,
                gazeVector.y,
                gazeVector.z,
                gazePoint.x,
                gazePoint.y,
                gazePoint.z,
                TagToInt(gazeObjectTag),
                pixelPositionObject.x,
                pixelPositionObject.y,
                pixelPositionObject.z,
            }
        }
        ;
        return new float[,]
        {
            {playerUp.x},
            {playerUp.y},
            {playerUp.z},
            {rightHandPos.x},
            {rightHandPos.y},
            {rightHandPos.z},
            {rightHandUp.x},
            {rightHandUp.y},
            {rightHandUp.z},
            {gazeVector.x},
            {gazeVector.y},
            {gazeVector.z},
            {gazePoint.x},
            {gazePoint.y},
            {gazePoint.z},
            {TagToInt(gazeObjectTag)},
            {pixelPositionObject.x},
            {pixelPositionObject.y},
            {pixelPositionObject.z}
        };
    }

    public float[,] GetData()
    {
        return new float[1,1];
    }

    public float[] GetFrame()
    {
        return frame;
    }

    /*public float[,] GetDataQueue()
    {
        var dataAsArray = new float[sequenceLength, inputLength];
        int seqIndex = 0;
        foreach (var li in loggedDataQueue)
        {
            for (int i = 0; i < inputLength; i++)
            {
                dataAsArray[seqIndex, i] = li[i];
            }

            seqIndex++;
        }

        return dataAsArray;
    }*/
    
    
}
