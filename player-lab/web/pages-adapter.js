const nativeFetch=globalThis.fetch.bind(globalThis);
const pagesBase=new URL(document.currentScript.dataset.base,location.origin);
const loaded=new Map();
async function localJSON(name){
 const cache=!/^20\d{2}$/.test(String(name));
 if(cache&&loaded.has(name))return loaded.get(name);
 const task=(async()=>{const r=await nativeFetch(new URL('_data/'+name+'.json.gz',pagesBase),{signal:AbortSignal.timeout(30000),credentials:'omit'});if(!r.ok)throw new Error('Published archive unavailable ('+r.status+').');if(typeof DecompressionStream==='undefined')throw new Error('This browser cannot decode the archive. Use a current browser.');return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json()})();
 if(cache)loaded.set(name,task);try{return await task}catch(e){loaded.delete(name);throw e}
}
let explorer,shots;
async function explorerHandler(){if(!explorer)explorer=Promise.all([localJSON('archive'),localJSON('saved-logs')]).then(([a,s])=>createHandler(a,{},s));try{return await explorer}catch(e){explorer=null;throw e}}
async function shotHandler(){if(!shots)shots=localJSON('shots').then(d=>{d.loadPack=localJSON;return createShotHandler(d)});try{return await shots}catch(e){shots=null;throw e}}
globalThis.fetch=async(input,options)=>{
 const u=new URL(typeof input==='string'||input instanceof URL?input:input.url,location.href);
 if(u.origin!==location.origin||!/^\/(api|legacy-api)\//.test(u.pathname))return nativeFetch(input,options);
 const request=new Request(u,input instanceof Request?input:options);
 if(request.signal.aborted)throw new DOMException('Request aborted','AbortError');
 const work=(async()=>{
  try{
   if(u.pathname.startsWith('/legacy-api/')){
    const result=await (await shotHandler())(request.clone());if(result)return result;
   }
   return await (await explorerHandler()).fetch(request,{},{});
  }catch(e){return new Response(JSON.stringify({error:e.message,detail:e.message}),{status:503,headers:{'Content-Type':'application/json'}})}
 })();
 return new Promise((resolve,reject)=>{const abort=()=>reject(new DOMException('Request aborted','AbortError'));request.signal.addEventListener('abort',abort,{once:true});work.then(resolve,reject).finally(()=>request.signal.removeEventListener('abort',abort))});
};
