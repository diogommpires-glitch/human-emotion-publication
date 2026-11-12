// Keep the original CommonJS entry point while testing the shipping module.
import("./memory.test.mjs").catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
