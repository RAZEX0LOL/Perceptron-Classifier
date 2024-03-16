class Neuron {
    constructor(weights, bias) {
        this.weights = weights;
        this.bias = bias;
    }

    activate(inputs) {
        let sum = this.bias;
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        return this.sigmoid(sum);
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
}

class Perceptron {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.hiddenLayer = [];
        for (let i = 0; i < hiddenSize; i++) {
            const weights = new Array(inputSize).fill(0).map(() => Math.random());
            const bias = Math.random();
            this.hiddenLayer.push(new Neuron(weights, bias));
        }

        this.outputLayer = [];
        for (let i = 0; i < outputSize; i++) {
            const weights = new Array(hiddenSize).fill(0).map(() => Math.random());
            const bias = Math.random();
            this.outputLayer.push(new Neuron(weights, bias));
        }
    }

    feedForward(inputs) {
        const hiddenOutputs = [];
        for (let neuron of this.hiddenLayer) {
            hiddenOutputs.push(neuron.activate(inputs));
        }

        const outputs = [];
        for (let neuron of this.outputLayer) {
            outputs.push(neuron.activate(hiddenOutputs));
        }

        return outputs;
    }

    train(inputs, targets, learningRate) {
        const hiddenOutputs = [];
        for (let neuron of this.hiddenLayer) {
            hiddenOutputs.push(neuron.activate(inputs));
        }

        const outputs = [];
        for (let neuron of this.outputLayer) {
            outputs.push(neuron.activate(hiddenOutputs));
        }

        const outputErrors = [];
        for (let i = 0; i < targets.length; i++) {
            outputErrors.push(targets[i] - outputs[i]);
        }

        const hiddenErrors = [];
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            let error = 0;
            for (let j = 0; j < this.outputSize; j++) {
                error += this.outputLayer[j].weights[i] * outputErrors[j];
            }
            hiddenErrors.push(error);
        }

        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.outputLayer[i].weights[j] += learningRate * outputErrors[i] * outputs[i] * (1 - outputs[i]) * hiddenOutputs[j];
            }
            this.outputLayer[i].bias += learningRate * outputErrors[i] * outputs[i] * (1 - outputs[i]);
        }

        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                this.hiddenLayer[i].weights[j] += learningRate * hiddenErrors[i] * hiddenOutputs[i] * (1 - hiddenOutputs[i]) * inputs[j];
            }
            this.hiddenLayer[i].bias += learningRate * hiddenErrors[i] * hiddenOutputs[i] * (1 - hiddenOutputs[i]);
        }

        // console.log("Output errors:", outputErrors);
        // console.log("Hidden errors:", hiddenErrors);
        // console.log("Output layer weights after training:", this.outputLayer.map(neuron => neuron.weights));
        // console.log("Hidden layer weights after training:", this.hiddenLayer.map(neuron => neuron.weights));
    }
}

function normalizeInputs(inputs) {
    const normalizedInputs = inputs.map((input, index) => {
        if (index === 0) {
            return input / 2025; // Нормализация года
        } else if (index === 1) {
            return input / 200000; // Нормализация пробега
        } else if (index === 2) {
            return input / 5000; // Нормализация массы
        } else if (index === 3) {
            return input / 10; // Нормализация объема двигателя
        }
    });
    return normalizedInputs;
}

const trainingData = [
    // Мотоцикл
    { inputs: [2020, 70000, 0.15, 0.2], targets: [0, 0, 1, 0] },
    { inputs: [2023, 40000, 0.2, 0.3], targets: [0, 0, 1, 0] },
    { inputs: [2015, 20000, 0.1, 0.15], targets: [0, 0, 1, 0] },
    { inputs: [2019, 60000, 0.18, 0.25], targets: [0, 0, 1, 0] },
    { inputs: [2022, 45000, 0.16, 0.22], targets: [0, 0, 1, 0] },

    // Внедорожник
    { inputs: [2020, 50000, 2.5, 3], targets: [0, 0, 0, 1] },
    { inputs: [2024, 150000, 4.0, 5], targets: [0, 0, 0, 1] },
    { inputs: [2025, 130000, 3.8, 4], targets: [0, 0, 0, 1] },
    { inputs: [2016, 160000, 4.2, 4.5], targets: [0, 0, 0, 1] },
    { inputs: [2018, 180000, 4.8, 5], targets: [0, 0, 0, 1] },

    // Легковой автомобиль
    { inputs: [2022, 80000, 1.8, 2], targets: [1, 0, 0, 0] },
    { inputs: [2021, 60000, 2.2, 3], targets: [1, 0, 0, 0] },
    { inputs: [2025, 50000, 1.8, 3], targets: [1, 0, 0, 0] },
    { inputs: [2010, 110000, 1.0, 1.5], targets: [1, 0, 0, 0] },
    { inputs: [2013, 140000, 1.2, 2], targets: [1, 0, 0, 0] },

    // Грузовик
    { inputs: [2023, 120000, 3.0, 5], targets: [0, 1, 0, 0] },
    { inputs: [2022, 70000, 3.5, 5], targets: [0, 1, 0, 0] },
    { inputs: [2024, 80000, 3.0, 5], targets: [0, 1, 0, 0] },
    { inputs: [2017, 180000, 4.5, 6], targets: [0, 1, 0, 0] },
    { inputs: [2019, 200000, 5.0, 7], targets: [0, 1, 0, 0] }
];

const inputSize = 4;
const hiddenSize = 8;
const outputSize = 4;

const perceptron = new Perceptron(inputSize, hiddenSize, outputSize);
const learningRate = 0.1;

for (let i = 0; i < 10000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    const inputs = normalizeInputs(data.inputs);
    const targets = data.targets;
    perceptron.train(inputs, targets, learningRate);
}

const testInput = [2010, 10000, 3, 10.5];
const normalizedTestInput = normalizeInputs(testInput);
const output = perceptron.feedForward(normalizedTestInput).map(value => Math.round(value));

const classNames = ["Легковой автомобиль", "Грузовик", "Мотоцикл", "Внедорожник"];
const predictedClassIndex = output.indexOf(Math.max(...output));
const predictedClassName = classNames[predictedClassIndex];

console.log("Test Input:", testInput);
console.log("Normalized Test Input:", normalizedTestInput);
console.log("Predicted Class:", predictedClassName);