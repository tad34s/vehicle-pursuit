using UnityEditor.SearchService;
using UnityEngine;

public class TrackGenerator : MonoBehaviour
{
	public GameObject marker;
	public GameObject checkpoints;

	public SplineCreator sc;

	public GameObject agent;
	private AgentCar carAgent;

	private GameObject currentRoad = null;

	public void Start()
	{
		sc.Init();

		carAgent = agent.GetComponent<AgentCar>();
	}

    public void Init()
    {
		// Debug.Log("Init track");

		RemoveTrack();

		currentRoad = GenerateTrack();
		Mesh roadMesh = currentRoad.GetComponent<MeshFilter>().mesh;

		RemoveMarkers();
		CreateMarkers(roadMesh);
		PlaceAgent(roadMesh);
    }

	private void RemoveTrack()
	{
		// Debug.Log("Removing track");
		if (currentRoad != null)
		{
			Destroy(currentRoad);
			currentRoad = null;
		}

		// Debug.Log("Removed track");
	}
	
	private void RemoveMarkers()
	{
		// Debug.Log("Removing markers");
		carAgent.setParentCheckpoint(null);
		foreach(Transform child in checkpoints.transform)
			Destroy(child.gameObject);
		// Debug.Log("Removed markers");
	}

    private void PlaceAgent(Mesh roadMesh)
	{
		// Debug.Log("Placing agent");

		Vector3[] vertices = roadMesh.vertices;
		float yLevel = 0.01f;

		Vector3 startingPosition = new Vector3(
			(vertices[0].x + vertices[1].x) / 2,
			yLevel,
			(vertices[0].z + vertices[1].z) / 2
		);

		Vector3 nextPos = new Vector3(
			(vertices[2].x + vertices[3].x) / 2,
			yLevel,
			(vertices[2].z + vertices[3].z) / 2
		);

		Vector3 targetDir = (nextPos - startingPosition);

		float angle = Vector3.SignedAngle(targetDir, Vector3.forward, Vector3.up);
		// Debug.Log("Angle: " + angle);
		Quaternion rotation = Quaternion.Euler(0, -angle, 0);

		agent.transform.position = startingPosition;
		agent.transform.rotation = rotation;

		// Debug.Log("Placed agent");
	}

	private GameObject GenerateTrack()
	{
		// Debug.Log("Generating track");
		sc.GenerateVoronoi();
		sc.GenerateSpline();
		GameObject track = sc.CreateTrack();

		// Debug.Log("Generated track");
		return track;
	}

	private void CreateMarkers(Mesh roadMesh)
	{
		// Debug.Log("Creating markers");
		// Debug.Log("Previous checkpoints: " + checkpoints.transform.childCount);
		Vector3[] vertices = roadMesh.vertices;
		for (var i = 0; i < vertices.Length - 3; i += 2)
		{
			Vector3 markerPosition = new Vector3(
				(vertices[i].x + vertices[i + 3].x) / 2,
				checkpoints.transform.position.y,
				(vertices[i].z + vertices[i + 3].z) / 2
			);
			Instantiate(marker, markerPosition, Quaternion.identity, checkpoints.transform);
		}

		// Debug.Log("Now checkpoints: " + checkpoints.transform.childCount);
		carAgent.setParentCheckpoint(checkpoints);

		// Debug.Log("Created markers");
	}
}
