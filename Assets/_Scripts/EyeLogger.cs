using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;
using Valve.VR;

public class EyeLogger : MonoBehaviour
{
    public bool logging;
    
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
    private string gazeObjectTag;

    public string objectInteractedWith = "";
    
    private static EyeLogger _instance;

    private Dictionary<String, int> nameToIntDict = new Dictionary<string, int>()
    {
        {"None", 0},
        {"WallShelf", 0},
        {"", 0},
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

    public static EyeLogger Instance { get { return _instance; } }

    private int logIndex;
    
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
        logIndex = PlayerPrefs.GetInt("Index", 0);
        logIndex++;
        PlayerPrefs.SetInt("Index", logIndex);
        currentFrame = 0;
        
        writer = new StreamWriter(GetPath());
        
        /*writer.WriteLine("Frame;" +
                         "Timestamp;" +
                         "Player Position;" +
                         "Player Rotation;" +
                         "Right Hand Position;" +
                         "Right Hand Rotation;" +
                         "Left Hand Position;" +
                         "Left Hand Rotation;" +
                         "Gaze Vector;" +
                         "Gaze Position;" +
                         "Gaze Object Tag;" + 
                         "Gaze Position Pixel Space"
                         );
*/
        string csv_header =
            "frame,time,player_pos_x,player_pos_y,player_pos_z,player_up_x,player_up_y,player_up_z,rel_r_hand_x,rel_r_hand_y,rel_r_hand_z,r_hand_up_x,r_hand_up_y,r_hand_up_z,gaze_vec_x,gaze_vec_y,gaze_vec_z,gaze_p_x,gaze_p_y,gaze_p_z,obj_tag,obj_interacted_with,gaze_to_screen_x,gaze_to_screen_y,gaze_to_screen_z,obj_interacted_with";
        
        writer.WriteLine(csv_header);
        
        //gaze position in pixel space
        //(dont collect yet) object position
    }
    
    private void OnApplicationQuit()
    {
        writer.Close();
    }
    
    void Update()
    {
        UpdateValues();

        if (Input.GetKeyDown(KeyCode.Space))
        {
            logging = !logging;

            if(logging)
                Debug.Log("Logging started!");
            else
                Debug.Log("Logging finished!");
        }

        if(!logging)
            return;

        currentFrame++;
    }

    public void Log(Vector3 gazeVec, Vector3 gazePt, string gazeObj)
    {
        gazeVector = gazeVec;
        gazePoint = gazePt;
        gazeObjectTag = gazeObj;
        
        //Debug.Log("Logged gaze object is: " + gazeObj);
        
        if(!logging)
            return;
        
        /*writer.WriteLine(currentFrame + ";" +
                         time + " " + time.Millisecond + ";" +
                         playerPosition.ToString("N4") + ";" +
                         playerRotation.ToString("N4") + ";" +
                         rightHandPosition.ToString("N4") + ";" +
                         rightHandRotation.ToString("N4") + ";" +
                         leftHandPosition.ToString("N4") + ";" +
                         leftHandRotation.ToString("N4") + ";" +
                         gazeVector.ToString("N4") + ";" +
                         gazePoint.ToString("N4") + ";" +
                         gazeObjectTag + ";" + //probably replace this with tagToInt
                         //TagToInt(gazeObjectTag) + ";" +
                         Camera.main.WorldToScreenPoint(gazePoint) + ";" //compare this to WorldToScreenVR. Not sure which one works or doesn't.
                         //WorldToScreenVR(Camera.main, gazePoint) + ";"
        );*/

        writer.WriteLine(GetLogAsString());
        
        objectInteractedWith = "";

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

    private Vector3 GetRelativePosition(Transform origin, Vector3 position)
    {
        Vector3 distance = position - origin.position;
        Vector3 relativePosition = Vector3.zero;
        relativePosition.x = Vector3.Dot(distance, origin.right.normalized);
        relativePosition.y = Vector3.Dot(distance, origin.up.normalized);
        relativePosition.z = Vector3.Dot(distance, origin.forward.normalized);

        return relativePosition;
    }

    private string VecToStr(Vector3 vec)
    {
        return vec.x.ToString("N4") + DELIM + vec.y.ToString("N4") + DELIM + vec.z.ToString("N4") + DELIM;
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
        //We focus only on right hand maybe?
        if (RELATIVE_POS)
        {
            output += VecToStr(GetRelativePosition(player.transform, rightHandPosition));
        }
        else
        {
            output += VecToStr(rightHandPosition);
        }

        output += VecToStr(rightHand.transform.up);
        
        //TODO: Figure out if rightHandRotation needs to be relative or not
        output += VecToStr(gazeVector) + VecToStr(gazePoint) + TagToInt(gazeObjectTag) + DELIM;
        output += VecToStr(Camera.main.WorldToScreenPoint(gazePoint));
        output += TagToInt(objectInteractedWith); // save when a player interacts with an object, not fed to the NN
        
        return output;

    }
    
    private string GetPath()
    {
#if UNITY_EDITOR
        return Application.dataPath + $"/eyeLog_{logIndex}.csv";
#endif
    }
    
    /// <summary>
    /// Calculates screen-space position a world space object. Useful for showing something on screen that is not visible in VR.
    /// For example, it can be used to update the position of a marker that highlights the gaze of the player, using eye tracking.
    /// </summary>
    /// <param name="camera">The camera used for VR rendering.</param>
    /// <param name="worldPos">World position of a point.</param>
    /// <returns>Screen position of a point.</returns>
    static Vector2 WorldToScreenVR(Camera camera, Vector3 worldPos)
    {
        Vector3 screenPoint = camera.WorldToViewportPoint(worldPos);
        float w = XRSettings.eyeTextureWidth;
        float h = XRSettings.eyeTextureHeight;
        float ar = w / h;

        screenPoint.x = (screenPoint.x - 0.15f * XRSettings.eyeTextureWidth) / 0.7f;
        screenPoint.y = (screenPoint.y - 0.15f * XRSettings.eyeTextureHeight) / 0.7f;

        return screenPoint;
    }

    private int TagToInt(string objectTag)
    {
        if (!nameToIntDict.ContainsKey(objectTag))
            throw new Exception($"{objectTag} not available in tags dictionary");

        return nameToIntDict[objectTag];
    }

    public void SetInteractedObject(string objectTag)
    {
        Debug.Log("Setting interacted object: " + objectTag);
        objectInteractedWith = objectTag;
    }
}
