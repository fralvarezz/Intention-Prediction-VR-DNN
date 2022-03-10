using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Valve.VR;

public class GameManager : MonoBehaviour
{
    public static GameManager instance;
    public Transform virtualWorld;
    private bool recenteredWorld;
    public SteamVR_TrackedObject trackedObject;
    public Transform offsetPoint;
    public GameObject cameraRig;
    public Camera mainCamera;
    private void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(this.gameObject);
        } 
        else 
        {
            instance = this;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.R))
            RecenterWorld();
    }

    public void RecenterWorld()
    {
        var cameraEuler = mainCamera.transform.eulerAngles;
        cameraEuler.x = 0;
        cameraEuler.z = 0;
        virtualWorld.transform.rotation = Quaternion.Euler(cameraEuler);
        
        Vector3 trackedObjectNewPos = trackedObject.transform.position;
        Vector3 difference = trackedObjectNewPos - offsetPoint.position;
        virtualWorld.position += difference;
        
        //Vector3 cameraRigPos = cameraRig.transform.position;
        //cameraRigPos.y += 0.51f;
        //cameraRig.transform.position = cameraRigPos;

        //virtualWorld.forward = mainCamera.transform.forward;
        
        recenteredWorld = true;
    }
}
