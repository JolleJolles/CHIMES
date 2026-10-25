// Run with Node.js and a local copy of the public reference module:
// node tests/fixtures/generate_boids_reference.mjs /tmp/Flocking-BjiUDGEy.js
// Source: https://geopatternlab.com/assets/Flocking-BjiUDGEy.js
// Execute the original step closure; do not reimplement its equations here.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import crypto from 'node:crypto';

const fixtureURL = new URL('./boids_reference.json', import.meta.url);
const fixture = JSON.parse(await fs.readFile(fixtureURL, 'utf8'));
const source = await fs.readFile(process.argv[2], 'utf8');
const start = source.indexOf('step:e=>{') + 5;
if (start < 5) throw new Error('Reference step closure not found');
const brace = source.indexOf('{', start);
let depth = 1, end = brace + 1;
for (; depth && end < source.length; end++) {
    if (source[end] === '{') depth++;
    else if (source[end] === '}') depth--;
}
if (depth) throw new Error('Unterminated step closure');
for (const scene of fixture.cases) {
    const t = {...fixture.params, perception: scene.perception ?? fixture.params.perception};
    const step = vm.runInNewContext('(' + source.slice(start, end) + ')',
        {t, r: 1000, w: 220, k: {current: {active: false}}});
    const state = {boids: scene.initial.map(([x,y,vx,vy]) => ({x,y,vx,vy,ax:0,ay:0,hist:[]}))};
    for (let i = 0; i < scene.steps; i++) step(state);
    scene.expected = state.boids.map(b => [b.x,b.y,b.vx,b.vy]);
}
fixture.sha256 = crypto.createHash('sha256').update(source).digest('hex');
await fs.writeFile(fixtureURL, JSON.stringify(fixture, null, 2) + '\n');
