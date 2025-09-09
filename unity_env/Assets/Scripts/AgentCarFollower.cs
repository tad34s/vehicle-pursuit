using System;
using System.Collections.Generic;
using System.Net.Http.Headers;
using Dreamteck.Splines;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AgentCarFollower : Agent
{
    public GameObject parentCheckpoint;

    public int currentCheckpoint = 0;

    private bool pauseLearning = false;

    const int k_Speed = 0;
    const int k_Steering = 1;

    public PrometeoCarControllerCont carController;

    public Camera carCamera;

    // public CarController carController;
    public Rigidbody rBody;
    public AgentCar carLeader;

    float deathPenalty = -10f;
    float bestDistance = 5f;

    public void Start()
    {
        carController.useControls = false;
        deathPenalty = DataChannel.getParameter("deathPenalty", -10f);
        bestDistance = DataChannel.getParameter("bestDistance", 5f);
    }

    public override void OnEpisodeBegin()
    {
        pauseLearning = true;
        pauseLearning = false;

        currentCheckpoint = 0;

        transform.position = transform.parent.position - new Vector3(0, 0, 5); ;
        transform.rotation = Quaternion.identity;


        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
    }


    private Vector2 GetRotatedRelativePosition()
    {
        float rotLeader = carLeader.transform.eulerAngles.y;
        float rotFollower = carLeader.transform.eulerAngles.y; ;

        Vector2 relativePosition = new Vector2(
                carLeader.transform.position.x - transform.position.x,
                carLeader.transform.position.z - transform.position.z
                );
        double neededRotation = rotFollower * Math.PI / 180;
        Vector2 rotatedRelativePosition = new Vector2(
                (float)(relativePosition.x * Math.Cos(neededRotation) - relativePosition.y * Math.Sin(neededRotation)),
                (float)(relativePosition.x * Math.Sin(neededRotation) + relativePosition.y * Math.Cos(neededRotation))
                );
        return rotatedRelativePosition;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // vehicle scalars
        sensor.AddObservation(carController.steeringAxis);
        sensor.AddObservation(carController.carSpeed);
        sensor.AddObservation(carLeader.carController.carSpeed);


        sensor.AddObservation(GetRotatedRelativePosition());

        // Calculate the signed angle (in degrees) around the Y-axis
        float signedAngle = Vector3.SignedAngle(
                    transform.rotation * Vector3.forward,
                    carLeader.transform.rotation * Vector3.forward,
                    Vector3.up
                );

        sensor.AddObservation(signedAngle);
    }

    public float calcDistanceToLeader()
    {
        Vector2 relativePosition = GetRotatedRelativePosition();
        float reward;
        if (relativePosition.y < 0)
        {
            reward = -relativePosition.magnitude;
        }
        else
        {
            float difference = relativePosition.magnitude - bestDistance;
            reward = (float)-0.01f * difference * difference + 10;
            reward *= relativePosition.y / relativePosition.magnitude;
        }



        float signedAngle = Vector3.SignedAngle(
                    transform.rotation * Vector3.forward,
                    carLeader.transform.rotation * Vector3.forward,
                    Vector3.up
                );

        reward += -10 * Math.Abs(signedAngle) / 180;
        return reward;
    }


    void TriggerAction(ActionBuffers actions)
    {
        float speed = actions.ContinuousActions[k_Speed];
        float steering = actions.ContinuousActions[k_Steering];

        carController.Movement(true, speed, steering);
        Debug.Log(speed);
        // Debug.Log(steering);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (pauseLearning)
            return;

        SetReward(calcDistanceToLeader());

        Debug.Log(carController.getAmountOfWheelsOnRoad());

        if (carController.getAmountOfWheelsOnRoad() <= 2)
        {
            SetReward(deathPenalty);
            carLeader.SetReward(carLeader.calcDistanceToCenter());
            EndEpisode();
            carLeader.EndEpisode();
        }

        AddReward((4 - carController.getAmountOfWheelsOnRoad()) * -1f);

        TriggerAction(actions);
        Debug.Log(carController.carSpeed);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[k_Speed] = 0;
        if (Input.GetKey(KeyCode.W))
            continuousActionsOut[k_Speed] += 1;

        if (Input.GetKey(KeyCode.S))
            continuousActionsOut[k_Speed] -= 1;

        continuousActionsOut[k_Steering] = 0;
        if (Input.GetKey(KeyCode.D))
            continuousActionsOut[k_Steering] += 1;

        if (Input.GetKey(KeyCode.A))
            continuousActionsOut[k_Steering] -= 1;
    }
}
