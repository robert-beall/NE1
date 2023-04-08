const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

// Define the model
const model = tf.sequential();
model.add(
  tf.layers.dense({ units: 10, inputShape: [2], activation: "sigmoid" })
);
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

// Compile the model
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Define the input data and expected output data
const input = tf.tensor2d([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const output = tf.tensor2d([[0], [1], [1], [0]]);

// Train the model
async function train() {
  await model.fit(input, output, { epochs: 10000 });
}

train();

// Make a prediction
const prediction = model.predict(tf.tensor2d([[0, 0]]));
//console.log(prediction.dataSync()[0]); // Output: a decimal number between 0 and 1
