using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReplayData
{
    public Vector3 playerHeadPosition { get; private set; }
    public Vector3 playerHeadUp { get; private set; }
    public Vector3 playerHeadForward { get; private set; }
    
    public Vector3 relativeControllerPosition { get; private set; }
    public Vector3 relativeControllerUp { get; private set; }
    public Vector3 relativeControllerForward { get; private set; }
    
    public Vector3 gazeVector { get; private set; }
    
    public int objectTag { get; private set; }

    public ReplayData(Vector3 playerHeadPosition, Vector3 playerHeadUp, Vector3 playerHeadForward,
        Vector3 relativeControllerPosition, Vector3 relativeControllerUp, Vector3 relativeControllerForward,
        Vector3 gazeVector, int objectTag)
    {
        this.playerHeadPosition = playerHeadPosition;
        this.playerHeadUp = playerHeadUp;
        this.playerHeadForward = playerHeadForward;
        this.relativeControllerPosition = relativeControllerPosition;
        this.relativeControllerUp = relativeControllerUp;
        this.relativeControllerForward = relativeControllerForward;
        this.gazeVector = gazeVector;
        this.objectTag = objectTag;
    }
}
