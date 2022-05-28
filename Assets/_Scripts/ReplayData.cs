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
    
    public Vector3 gazePoint { get; private set; }
    
    public Vector2 gazePointToScreen { get; private set; }
    
    public int objectTag { get; private set; }

    public ReplayData(Vector3 playerHeadPosition, Vector3 playerHeadUp, Vector3 playerHeadForward,
        Vector3 relativeControllerPosition, Vector3 relativeControllerUp, Vector3 relativeControllerForward,
        Vector3 gazeVector, Vector3 gazePoint, Vector3 gazePointToScreen, int objectTag)
    {
        this.playerHeadPosition = playerHeadPosition;
        this.playerHeadUp = playerHeadUp;
        this.playerHeadForward = playerHeadForward;
        this.relativeControllerPosition = relativeControllerPosition;
        this.relativeControllerUp = relativeControllerUp;
        this.relativeControllerForward = relativeControllerForward;
        this.gazeVector = gazeVector;
        this.gazePoint = gazePoint;
        this.gazePointToScreen = gazePointToScreen;
        this.objectTag = objectTag;
    }
    
    public float[] GetData()
    {
        return new float[] {
            //playerHeadPosition.x, playerHeadPosition.y, playerHeadPosition.z,
            playerHeadUp.x, playerHeadUp.y, playerHeadUp.z,
            playerHeadForward.x, playerHeadForward.y, playerHeadForward.z,
            relativeControllerPosition.x, relativeControllerPosition.y, relativeControllerPosition.z,
            relativeControllerUp.x, relativeControllerUp.y, relativeControllerUp.z,
            relativeControllerForward.x, relativeControllerForward.y, relativeControllerForward.z,
            gazeVector.x, gazeVector.y, gazeVector.z,
            gazePoint.x, gazePoint.y, gazePoint.z,
            gazePointToScreen.x, gazePointToScreen.y,
            objectTag
        };
    }
}
