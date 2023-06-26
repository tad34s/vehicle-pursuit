using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Net.Mime;

public class AgentCar : Agent
{
	public GameObject parentCheckpoint;
	List<GameObject> checkpoints = new List<GameObject>();
	public int currentCheckpoint = 0;

	private Quaternion startingRotation;

	const int k_Forward = 0;
	const int k_Back = 1;
	const int k_Left = 1;
	const int k_Right = 2;

	private PrometeoCarController carController;
	private Rigidbody rBody;

    public void Start()
    {
		carController = GetComponent<PrometeoCarController>();
		carController.useControls = false;

		rBody = GetComponent<Rigidbody>();

		startingRotation = transform.rotation;

		for(int i = 0; i < parentCheckpoint.transform.childCount; i++)
		{
			checkpoints.Add(parentCheckpoint.transform.GetChild(i).gameObject);
		}
    }

    public override void OnEpisodeBegin()
    {
		transform.position = new Vector3(0f, 0.1f, 0f);
		transform.rotation = startingRotation;

		rBody.velocity = Vector3.zero;
		rBody.angularVelocity = Vector3.zero;

		currentCheckpoint = 0;

		Debug.Log("New episode");
    }

    public override void CollectObservations(VectorSensor sensor)
    {
		sensor.AddObservation(carController.carSpeed);
		sensor.AddObservation(carController.steeringAxis);

		sensor.AddObservation(calcDistanceToNextCheckpoint());
    }

	private GameObject getNextCheckpoint()
	{
		if(currentCheckpoint + 1 >= checkpoints.Count)
			return checkpoints[0];
		return checkpoints[currentCheckpoint + 1];
	}
	private GameObject getPreviousCheckpoint()
	{
		if(currentCheckpoint - 1 < 0)
			return checkpoints[checkpoints.Count - 1];
		return checkpoints[currentCheckpoint - 1];
	}

	private float calcDistance(Vector3 pos1, Vector3 pos2)
	{
		return Vector2.Distance(
			new Vector2(pos1.x, pos1.z),
			new Vector2(pos2.x, pos2.z)
		);
	}

	private float calcDistanceToNextCheckpoint()
	{
		Vector3 nextCheckpointPosition = getNextCheckpoint().transform.position;
		return calcDistance(nextCheckpointPosition, transform.position);
	}
	private float calcDistanceToPreviousCheckpoint()
	{
		Vector3 previousCheckpointPosition = getPreviousCheckpoint().transform.position;
		return calcDistance(previousCheckpointPosition, transform.position);
	}

	private float getDrivenDistance()
	{
		float distance = 0f;

		for(int i = 0; i < currentCheckpoint; i++)
		{
			distance += calcDistance(
				checkpoints[i].transform.position,
				checkpoints[i + 1].transform.position
			);
		}

		distance += calcDistance(
			checkpoints[checkpoints.Count - 1].transform.position,
			transform.position
		);

		return distance;
	}

    void TriggerAction(ActionBuffers actions)
	{
		bool goForward = actions.DiscreteActions[k_Forward] == 1;
		// bool goForward = true;
		// bool goBack = actions.DiscreteActions[k_Back] == 1;
		bool goBack = false;
		bool turnLeft = actions.DiscreteActions[k_Left] == 1;
		bool turnRight = actions.DiscreteActions[k_Right] == 1;
		// Debug.Log($"Forward: {goForward}\nBackward: {goBack}\nLeft: {turnLeft}\nRight: {turnRight}");

		if (goForward)
		{
			carController.GoForwardAction();
		}

		if (goBack)
		{
			carController.GoReverseAction();
		}

		if (turnLeft)
		{
			carController.TurnLeftAction();
		}
		if (turnRight)
		{
			carController.TurnRightAction();
		}
		if (!turnLeft && !turnRight)
		{
			carController.ThrottleOffAction();
		}
		if (!goBack && !goForward && !carController.deceleratingCar)
		{
			carController.DecelerateCarAction();
		}
		if (!turnLeft && !turnRight && carController.steeringAxis != 0f)
		{
			carController.ResetCarSteeringAction();
		}
	}

    public override void OnActionReceived(ActionBuffers actions)
    {
		if(calcDistanceToNextCheckpoint() < 3f)
		{
            AddReward(5f);
            currentCheckpoint++;
		}

		if (carController.getAmountOfWheelsOnRoad() <= 3)
		{
			Debug.Log("Tire on terrain. Resetting");
			SetReward(-1f);
			EndEpisode();
		}

		// SetReward(carController.getAmountOfWheelsOnRoad() * 0.0001f);
		// SetReward(4 - carController.getAmountOfWheelsOnRoad() * -0.1f);


        if(carController.carSpeed > 2f)
        {
            float reward = getDrivenDistance() * 0.5f;
            Debug.Log(reward);
            AddReward(reward);
        } else
		{
			AddReward(-1f);
		}

		TriggerAction(actions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
		var discreteActionsOut = actionsOut.DiscreteActions;

		discreteActionsOut[k_Forward] = Input.GetKey(KeyCode.W) ? 1 : 0;
		// discreteActionsOut[k_Back] = Input.GetKey(KeyCode.S) ? 1 : 0;
		discreteActionsOut[k_Left] = Input.GetKey(KeyCode.A) ? 1 : 0;
		discreteActionsOut[k_Right] = Input.GetKey(KeyCode.D) ? 1 : 0;
    }
}