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

    const int k_Forward = 0;
    const int k_Back = 1;
    const int k_Left = 2;
    const int k_Right = 3;

    public PrometeoCarController carController;

    public Camera carCamera;

    // public CarController carController;
    public Rigidbody rBody;
    public AgentCar carLeader;

    float deathPenalty = -10f;
    float bestDistance = 10f;

    public void Start()
    {
        carController.useControls = false;
        deathPenalty = DataChannel.getParameter("deathPenalty", -10f);
        bestDistance = DataChannel.getParameter("bestDistance", 10f);
    }

    public override void OnEpisodeBegin()
    {
        pauseLearning = true;
        pauseLearning = false;
 
        currentCheckpoint = 0;
 
        transform.position = transform.parent.position - new Vector3(0, 0, 5);;
        transform.rotation = Quaternion.identity;
 
        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(true);
        sensor.AddObservation(carController.steeringAxis);
        sensor.AddObservation(carController.carSpeed);

        sensor.AddObservation(carLeader.carController.carSpeed);
        sensor.AddObservation(carLeader.transform.position);
        sensor.AddObservation(carLeader.transform.rotation);
    }

    float calcDistanceToLeader()
    {
        float difference = bestDistance - Vector3.Distance(transform.position, carLeader.transform.position);

        return difference * difference;
    }

    void TriggerAction(ActionBuffers actions)
    {
        bool goForward = actions.DiscreteActions[k_Forward] == 1;
        // bool goForward = true;
        bool goBack = actions.DiscreteActions[k_Back] == 1;
        // bool goBack = false;
        bool turnLeft = actions.DiscreteActions[k_Left] == 1;
        bool turnRight = actions.DiscreteActions[k_Right] == 1;
 
        carController.Movement(true, goForward, goBack, turnLeft, turnRight);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (pauseLearning)
            return;

        SetReward(calcDistanceToLeader());

        if (carController.getAmountOfWheelsOnRoad() <= 2)
        {
            SetReward(deathPenalty);
            EndEpisode();
            carLeader.EndEpisode();
        }

        AddReward((4 - carController.getAmountOfWheelsOnRoad()) * -1f);

        TriggerAction(actions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        discreteActionsOut[k_Forward] = Input.GetKey(KeyCode.W) ? 1 : 0;
        discreteActionsOut[k_Back] = Input.GetKey(KeyCode.S) ? 1 : 0;
        discreteActionsOut[k_Left] = Input.GetKey(KeyCode.A) ? 1 : 0;
        discreteActionsOut[k_Right] = Input.GetKey(KeyCode.D) ? 1 : 0;
    }
}
