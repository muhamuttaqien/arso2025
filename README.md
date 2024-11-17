# arso2025
Our codebase for publishing research work to [ARSO Conference](https://ieee-arso.org/), July 17th-19th 2025, Osaka, Japan

## Background

The problem we aim to address in this research is how to enhance interaction between humans and robots in a smart home setting, particularly in the kitchen, with a focus on a robotic arm manipulator. We believe this can be achieved by implementing a multimodal robot that can understand both visual images and natural language instructions from humans to assist with tasks such as food preparation, utensil handling, and other kitchen activities.

The methodology we propose involves using end-to-end deep reinforcement learning to process multimodal data (images and language), while enhancing the learning process through effective curriculum learning. The training environment will be simulated using the [AI2Thor simulator](https://ai2thor.allenai.org/manipulathor/), with the robot's capabilities focused solely on the manipulation tasks of an arm manipulator within this context.

## Kitchen-Specific Curriculum with Language Complexity

This curriculum leverages natural language instructions to gradually increase the complexity of manipulation tasks in a simulated AI2-THOR kitchen environment. The progression is designed to help the agent acquire foundational skills before tackling advanced tasks involving spatial reasoning, filtering, and multi-step instructions.

### Curriculum Levels

| **Level** | **Natural Language Instruction**                      | **Task Description**                                                                                     | **Environment Complexity**                                                                                                    |
|-----------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| 1         | "Pick up the red mug."                                | Single-step task: Grasp a single object (e.g., red mug).                                                 | Mug placed on an easily accessible counter with no distractors.                                                               |
| 2         | "Pick up the red mug on the kitchen counter."          | Grasp a single object at a specific location (e.g., "on the counter").                                   | Add one or two other objects (e.g., plate, spoon) for variety, but no overlap or ambiguity.                                   |
| 3         | "Pick up the red mug and avoid the blue mug."          | Grasp the correct object while ignoring a distractor (e.g., blue mug).                                   | Place both objects in close proximity to introduce subtle distractions.                                                      |
| 4         | "Pick up the red mug and place it in the sink."        | Two-step task: Grasp the target object and place it in the specified location (e.g., sink).               | Add spatial reasoning by placing the sink away from the counter.                                                             |
| 5         | "Pick up the red mug on the counter and place it on the dining table." | Multi-step task: Identify the object, pick it up, and place it in a new location (e.g., dining table).     | Introduce a random placement for the dining table and counter to vary task difficulty.                                       |
| 6         | "Pick up the smallest red mug."                       | Object filtering: Select the target object based on multiple attributes (color, size).                   | Place multiple mugs of varying sizes and colors on the counter.                                                              |
| 7         | "Pick up the red mug from the counter and avoid obstacles to place it in the sink." | Advanced reasoning: Avoid obstacles while completing the multi-step task.                                | Add obstacles like other objects, chairs, or random clutter in the path to the sink.                                         |
| 8         | "Pick up the red mug and place it in the nearest cabinet to the fridge." | Hierarchical reasoning: Identify the correct placement location based on a spatial relationship.          | Add multiple cabinets with varying distances from the fridge.                                                                |

### How It Works

1. **Task Progression**:
   - The agent begins with simple tasks, such as grasping a single object, and progresses to multi-step tasks requiring spatial reasoning and filtering.
2. **Dynamic Environment**:
   - Each task is configured dynamically using AI2-THOR to place objects and adjust complexity based on the level.
3. **Language Integration**:
   - Natural language instructions are provided to guide the agent and vary in complexity to match the task.

### Example Usage

- **Level 1**: "Pick up the red mug." — The agent identifies a red mug on a counter and grasps it.
- **Level 4**: "Pick up the red mug and place it in the sink." — The agent must grasp the red mug and place it in the sink, which may require spatial reasoning.
- **Level 8**: "Pick up the red mug and place it in the nearest cabinet to the fridge." — The agent identifies the nearest cabinet and completes a multi-step hierarchical task.

---

### Notes

- The curriculum is designed for use with AI2-THOR kitchen environments (e.g., `FloorPlan1` to `FloorPlan30`).
- The complexity progression ensures the agent learns basic manipulation skills before tackling advanced reasoning tasks.
- Tasks can be customized further based on specific research goals or scenarios.
